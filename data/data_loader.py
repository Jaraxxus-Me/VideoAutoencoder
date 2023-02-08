import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import os
import json
import yaml

with open('configs/datasets.yaml') as f:
    config_data = f.read()
    config = yaml.safe_load(config_data)

def image_loader(path, input_size):
    image = cv2.imread(path)
    input_h, input_w = input_size, input_size
    image = cv2.resize(image, (input_w, input_h))
    return image

def rgb_preprocess(images):
    images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
    images = torch.stack([transforms.ToTensor()(image) for image in images])
    return images

def masked_rgb_preprocess(images, masks):
    images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
    images = torch.stack([transforms.ToTensor()(image) for image in images])
    masks = torch.stack([transforms.ToTensor()(mask) for mask in masks])
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
    def __init__(self, data, dataset, is_train=True):
        self.imagefiles = data
        self.is_train = is_train
        self.dataset = dataset
        # build gt trajectory
        with open (config[self.dataset]['cam_path'], 'r') as f:
            self.cam_para = json.load(f)

    def __getitem__(self, index):
        # imgs
        imageset = self.imagefiles[index]
        maskset = [os.path.join(config[self.dataset]['mask_path'], f.split('/')[-1]) for f in imageset]
        input_size = config[self.dataset]['input_size']
        images = [image_loader(file, input_size) for file in imageset]
        masks = [image_loader(file, input_size) for file in maskset]
        images_rgb = masked_rgb_preprocess(images, masks)
        # poses
        imgids = [int(f.split('/')[-1][0:-4]) for f in imageset]
        rela_traj = cam2pose_rel(self.cam_para, imgids)

        return images_rgb, rela_traj

    def __len__(self):
        return len(self.imagefiles)
