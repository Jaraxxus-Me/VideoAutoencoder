import argparse

def train_parser():
    parser = argparse.ArgumentParser(description='Video Auto-encoder Training Options')

    # experiment specifics
    parser.add_argument('--dataset', default='OWID',
                        help='Name of the dataset.')
    parser.add_argument('--savepath', type=str, default='log/train_local/Try8/OWID_3/',
                        help='Path for checkpoints and logs')
    parser.add_argument('--resume', type=str, default='log/train_local/Try8/OWID_2/checkpoint_best.tar',
                        help='Checkpoint file to resume')
    parser.add_argument('--worker', type=int, default=16,
                        help='number of dataloader threads')

    # for displays
    parser.add_argument('--valid_freq', type=int, default=1000,
                        help='frequency of validation')


    # model options
    parser.add_argument('--encoder_3d', type=str, default='conv',
                        help='3d encoder architecture')
    parser.add_argument('--encoder_traj', type=str, default='conv',
                        help='trajectory encoder architecture')
    parser.add_argument('--decoder', type=str, default='conv',
                        help='decoder style')
    parser.add_argument('--padding_mode', type=str, default='zeros',
                        help='grid sampling padding mode')

    # training options
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train')
    parser.add_argument('--bsize', type=int, default=4,
                        help='batch size for training')

    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate')
    parser.add_argument('--d_lr', type=float, default=1e-4,
                        help='learning rate for gan discriminator')

    parser.add_argument('--lambda_l1', type=float, default=10,
                        help='re-weight l1 loss')
    parser.add_argument('--lambda_perc', type=float, default=0.1,
                        help='re-weight perceptual loss')
    parser.add_argument('--lambda_gan', type=float, default=0.01,
                        help='re-weight gan loss')
    # parser.add_argument('--lambda_voxel', type=float, default=0.0,
    #                     help='re-weight voxel consistency loss')

    parser.add_argument('--interval', type=int, default=2,
                        help='interval between clip frames')
    parser.add_argument('--clip_length', type=int, default=6,
                        help='number of frames in the training clip')
    parser.add_argument('--lr_adj', type=float, default=1.0,
                        help='a multiplier to adjust the training schedule')

    return parser


