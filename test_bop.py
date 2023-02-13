import argparse
import os
import time
import logger
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.io as io
from parser import test_bop_parser
import data.image_folder as D
import data.data_loader as DL
from models.autoencoder import *
from test_helper import *
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

parser = test_bop_parser()
args = parser.parse_args()
np.set_printoptions(precision=3)

def gettime():
    # get GMT time in string
    return time.strftime("%Y%m%d%H%M%S", time.gmtime())

def main():
    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
    # args.savepath = args.savepath+f'/test_re10k_{gettime()}'
    log = logger.setup_logger(args.savepath + '/testing.log')

    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    _, TestData = D.random_dataloader(args.dataset, 6, n_valid=30,
                               is_train=args.train_set, load_all_frames=True)
    TestLoader = DataLoader(DL.MaskedImageFloder(TestData, args.dataset, log, is_train=False),
                            batch_size=1, shuffle=False, num_workers=0)

    # get auto-encoder
    encoder_3d = Encoder3D(args)
    rotate = Rotate(args)
    rotate_inv = RotateInv(args)
    decoder = Decoder(args)

    # cuda
    encoder_3d = nn.DataParallel(encoder_3d).cuda()
    rotate = nn.DataParallel(rotate).cuda()
    rotate_inv = nn.DataParallel(rotate_inv).cuda()
    decoder = nn.DataParallel(decoder).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            encoder_3d.load_state_dict(checkpoint['encoder_3d'])
            decoder.load_state_dict(checkpoint['decoder'])
            rotate.load_state_dict(checkpoint['rotate'])
            rotate_inv.load_state_dict(checkpoint['rotate_inv'])
            log.info("=> loaded checkpoint '{}'".format(args.resume))
        else:
            log.info("=> No checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('=> No checkpoint file. Start from scratch.')

    start_full_time = time.time()
    with torch.no_grad():
        log.info('start testing.')
        test(TestData, TestLoader, encoder_3d, decoder, rotate, rotate_inv, log)
    log.info('full testing time = {:.2f} Minutes'.format((time.time() - start_full_time) / 60))

def test(data, dataloader, encoder_3d, decoder, rotate, rotate_inv, log):
    _loss = AverageMeter()
    video_limit = min(args.video_limit, len(dataloader))
    frame_limit = args.frame_limit
    for b_i, data in tqdm(enumerate(dataloader)):
        if b_i == video_limit: break

        encoder_3d.eval()
        decoder.eval()
        rotate.eval()

        input_ids = data[0][0]
        input_clip = data[1][0].cuda()
        gt_traj = data[2][0].cuda()
        target_clip = data[3][0].cuda()
        t, c, h, w = input_clip.size()
        n = target_clip.shape[0]

        preds = []
        for i in range(n):
            if i == 0:
                # preds.append(input_clip[0:1])
                scene_rep = encoder_3d(input_clip)
                H, W, D = scene_rep.shape[2], scene_rep.shape[3], scene_rep.shape[4]
                # scene_index = 0
                # affine
                theta = gt_traj[n:].reshape(t, 3, 4)
                rot_codes_inv = rotate(scene_rep, theta).view(1, t, -1, H, W, D)
                # aggregate
                rot_codes_inv = rot_codes_inv.mean(dim=1, keepdim=True).view(1, -1, H, W, D)
            # elif i % args.reinit_k == 0:
            #     # reinitialize 3d voxel
            #     scene_rep = encoder_3d(pred)
            #     scene_index = i
            theta = gt_traj[:n][i].unsqueeze(0)
            rot_codes_local = rotate_inv(rot_codes_inv, theta)
            # decode
            output = decoder(rot_codes_local)
            pred = F.interpolate(output, (h, w), mode='bilinear')
            pred = torch.clamp(pred, 0, 1)
            preds.append(pred)

        # output
        synth_save_dir = os.path.join(args.savepath, f"Videos")
        os.makedirs(synth_save_dir, exist_ok=True)
        preds = torch.cat(preds,dim=0)
        pred = (preds.permute(0,2,3,1) * 255).byte().cpu()
        io.write_video(synth_save_dir+f'/video_{b_i}_pred.mp4', pred, 6)
        vid = (target_clip.permute(0,2,3,1) * 255).byte().cpu()
        io.write_video(synth_save_dir+f'/video_{b_i}_true.mp4', vid, 6)
    print()

if __name__ == '__main__':
    main()