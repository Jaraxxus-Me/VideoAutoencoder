import torch
import numpy as np
from glob import glob
from tqdm import tqdm
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import os
import json
import yaml

IMG_EXT = ['png', 'jpg']

with open('configs/datasets.yaml') as f:
    config_data = f.read()
    config = yaml.safe_load(config_data)

def image_loader(path, input_size):
    image = cv2.imread(path)
    input_h, input_w = input_size, input_size
    image = cv2.resize(image, (input_w, input_h))
    return image

def cal_objsz(mask_path):
    mask_paths = sum([sorted(glob(mask_path+f'/*.{ext}')) for ext in IMG_EXT],[])
    obj_sz = 0
    for mask_path in mask_paths:
        mask = cv2.imread(mask_path)
        W, H = mask.shape[0], mask.shape[1]
        # mask to box:
        x, y = np.where(mask[:,:,0] != 0)
        obj_w = np.max(x) - np.min(x)
        obj_h = np.max(y) - np.min(y)
        local_obj_sz = np.max([obj_w, obj_h]) + 5
        local_obj_sz = np.min([local_obj_sz, W, H])
        if local_obj_sz>obj_sz:
            obj_sz = local_obj_sz
    return obj_sz

def crop_image_loader(rgb_path, mask_path, obj_sz, input_size):
    rgb = cv2.imread(rgb_path)
    mask = cv2.imread(mask_path)
    W, H = rgb.shape[0], rgb.shape[1]
    left_x = (W - obj_sz)//2
    left_y = (H - obj_sz)//2
    # crop obj center
    rgb = rgb[left_x:left_x+obj_sz, left_y:left_y+obj_sz, :]
    mask = mask[left_x:left_x+obj_sz, left_y:left_y+obj_sz, :]
    # resize
    input_h, input_w = input_size, input_size
    image = cv2.resize(rgb, (input_w, input_h))
    mask = cv2.resize(mask, (input_w, input_h))
    return image, mask

def rgb_preprocess(images):
    images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
    images = torch.stack([transforms.ToTensor()(image) for image in images])
    return images

def masked_rgb_preprocess(mask_images):
    images = [cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB) for image in mask_images]
    masks = torch.stack([transforms.ToTensor()(image[1]) for image in mask_images])
    images = torch.stack([transforms.ToTensor()(image) for image in images])
    images = images*masks
    return images

def cam2pose_rel(cam, ids):
    # relative to the first frame
    rela_r = torch.eye(3).unsqueeze(0).repeat(len(ids), 1, 1)
    rela_t = torch.zeros((len(ids), 3, 1))
    pre_r = torch.Tensor(cam[str(ids[0])]["cam_R_w2c"]).reshape(3, 3)
    pre_t = torch.Tensor(cam[str(ids[0])]["cam_t_w2c"]).reshape(3, 1)
    for i, img_id in enumerate(ids):
        curr_r = torch.Tensor(cam[str(img_id)]["cam_R_w2c"]).reshape(3, 3)
        curr_t = torch.Tensor(cam[str(img_id)]["cam_t_w2c"]).reshape(3, 1)
        rela_r[i] = curr_r @ pre_r.T
        # rela_t[i, :] = curr_t - pre_t
    rela_pose_1 = torch.cat([rela_r, rela_t], dim=-1)
    # relative to the first frame, inverse
    rela_r = torch.eye(3).unsqueeze(0).repeat(len(ids), 1, 1)
    rela_t = torch.zeros((len(ids), 3, 1))
    pre_r = torch.Tensor(cam[str(ids[0])]["cam_R_w2c"]).reshape(3, 3)
    pre_t = torch.Tensor(cam[str(ids[0])]["cam_t_w2c"]).reshape(3, 1)
    for i, img_id in enumerate(ids):
        curr_r = torch.Tensor(cam[str(img_id)]["cam_R_w2c"]).reshape(3, 3)
        curr_t = torch.Tensor(cam[str(img_id)]["cam_t_w2c"]).reshape(3, 1)
        rela_r[i] = (curr_r @ pre_r.T).T
        # rela_t[i, :] = curr_t - pre_t
    rela_pose_t = torch.cat([rela_r, rela_t], dim=-1)
    # both input
    rela_pose = torch.cat([rela_pose_1, rela_pose_t], dim=0)
    return rela_pose

class ImageFloder(data.Dataset):
    def __init__(self, data, dataset, is_train=True):
        self.imagefiles = data
        self.is_train = is_train
        self.dataset = dataset

    def __getitem__(self, index):
        imageset = self.imagefiles[index]
        input_size = config[self.dataset]['input_size']
        images = [image_loader(file, input_size) for file in imageset]
        images_rgb = rgb_preprocess(images)

        return images_rgb

    def __len__(self):
        return len(self.imagefiles)

class MaskedImageFloder(data.Dataset):
    def __init__(self, data, dataset, log, is_train=True):
        self.imagefiles = data
        self.is_train = is_train
        self.dataset = dataset
        if os.path.isfile(config[self.dataset]['sz_path']):
            log.info('Loading object camera and size from existing path')
            self.obj_sz = np.load(config[self.dataset]['sz_path'], allow_pickle=True)['arr_0'].item()
            with open (config[self.dataset]['cam_path'], 'r') as f:
                self.cam = json.load(f)
        else:
            log.info('Start constructing object camera and size')
            self.cam = {}
            self.obj_sz = {}
            if self.is_train:
                video_paths = sorted(glob(config[self.dataset]['train_path']+'/*/'))
            else:
                video_paths = sorted(glob(config[self.dataset]['test_path']+'/*/'))
            for v in video_paths:
                obj_id = v.split('/')[-2]
                log.info('OBJ {}/{}'.format(obj_id, len(video_paths)))
                cam_path = os.path.join(v, 'scene_camera.json')
                with open (cam_path, 'r') as f:
                    self.cam[obj_id] = json.load(f)
                maskset = os.path.join(v, 'mask')
                self.obj_sz[obj_id] = cal_objsz(maskset)
            log.info('Constructing object camera and size done!')
            np.savez(config[self.dataset]['sz_path'], self.obj_sz)
            with open (config[self.dataset]['cam_path'], 'w') as f:
                json.dump(self.cam, f)

    def __getitem__(self, index):
        # imgs
        imageset = self.imagefiles[index]
        obj_id = imageset[0].split('/')[-3]
        maskset = [f.replace('rgb', 'mask') for f in imageset]
        input_size = config[self.dataset]['input_size']
        mask_images = [crop_image_loader(imageset[img_id], maskset[img_id], self.obj_sz[obj_id], input_size) for img_id in range(len(imageset))]
        images_rgb = masked_rgb_preprocess(mask_images)
        # poses
        imgids = [int(f.split('/')[-1][0:-4]) for f in imageset]
        rela_traj = cam2pose_rel(self.cam[obj_id], imgids)

        return images_rgb, rela_traj

    def __len__(self):
        return len(self.imagefiles)
