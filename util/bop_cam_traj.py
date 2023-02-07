import numpy as np
import os
import json
import open3d as o3d
import imageio.v3 as iio
from tqdm import tqdm

# path config
base_path = '/home/airlab/storage/airlab/data_0108/data/InsDet/ZID-1M/P1'
seqs = os.listdir(base_path)
for seq in seqs:
    seq_path = os.path.join(base_path, seq)
    if not os.path.isdir(seq_path):
        continue
    print("start seq: {}".format(seq))
    seq_path = os.path.join(base_path, seq)
    cam_path = os.path.join(seq_path, 'scene_camera.json')

    # camera par
    with open (cam_path, 'r') as f:
        cam_para = json.load(f)

    for img_id in tqdm(cam_para.keys()):
        R = np.array(cam_para[img_id]["cam_R_w2c"]).reshape(3, 3)
        T = np.array(cam_para[img_id]["cam_t_w2c"]).reshape(3, 1)