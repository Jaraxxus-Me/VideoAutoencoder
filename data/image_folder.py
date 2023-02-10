import yaml
import os
from glob import glob
import numpy as np

with open('configs/datasets.yaml') as f:
    config_data = f.read()
    config = yaml.safe_load(config_data)

IMG_EXT = ['png', 'jpg']

def dataloader(dataset, clip_length, interval, n_valid=10, is_train=True, load_all_frames=False):
    if is_train:
        data_path = config[dataset]['train_path']
    else:
        data_path = config[dataset]['test_path']

    if data_path is None:
        raise RuntimeError(f"dataset does not support {'train' if is_train else 'test'} mode.")

    video_paths = sorted(glob(data_path+'/*/'))
    video_paths = [os.path.join(v_path, 'rgb') for v_path in video_paths]
    batches_train, batches_valid = [], []
    for index in range(len(video_paths)):
        vpath = video_paths[index]
        fnames = sum([sorted(glob(vpath+f'/*.{ext}')) for ext in IMG_EXT],[])
        fnames = fnames[::interval]

        video_batches = []
        if load_all_frames:
            video_batches.append(fnames)
        else:
            while 1:
                if len(fnames) < clip_length:
                    break

                frame_sequence = fnames[:clip_length]
                video_batches.append(frame_sequence)
                fnames = fnames[1:]  # skip first one

        if index >= n_valid:
            batches_train.extend(video_batches)
        else:
            batches_valid.extend(video_batches)

    return batches_train, batches_valid

def random_dataloader(dataset, clip_length, n_valid=10, n_subvideo=20, is_train=True, load_all_frames=False):
    if is_train:
        data_path = config[dataset]['train_path']
    else:
        data_path = config[dataset]['test_path']

    if data_path is None:
        raise RuntimeError(f"dataset does not support {'train' if is_train else 'test'} mode.")

    video_paths = sorted(glob(data_path+'/*/'))
    video_paths = [os.path.join(v_path, 'rgb') for v_path in video_paths]
    batches_train, batches_valid = [], []
    for index in range(len(video_paths)):
        vpath = video_paths[index]
        fnames = sum([sorted(glob(vpath+f'/*.{ext}')) for ext in IMG_EXT],[])
        # for one base video
        video_batches = []
        if load_all_frames:
            img_ids = np.random.choice(len(fnames), clip_length, replace=False)
            frame_sequence = [fnames[img_id] for img_id in img_ids]
            video_batches.append(frame_sequence)
        else:
            while len(video_batches) < n_subvideo:
                img_ids = np.random.choice(len(fnames), clip_length, replace=False)
                img_ids.sort()
                frame_sequence = [fnames[img_id] for img_id in img_ids]
                video_batches.append(frame_sequence)

        if index >= n_valid:
            batches_train.extend(video_batches)
        else:
            batches_valid.extend(video_batches)

    return batches_train, batches_valid
