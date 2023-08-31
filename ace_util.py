# Copyright © Niantic, Inc. 2022.
from pathlib import Path

import cv2
import numpy as np
import skimage
import torch
import json
from pykdtree.kdtree import KDTree
from scipy.spatial.transform import Rotation
from skimage.transform import resize as ski_resize
from skimage.transform import rotate as ski_rotate


def get_pixel_grid(subsampling_factor):
    """
    Generate target pixel positions according to a subsampling factor, assuming prediction at center pixel.
    """
    pix_range = torch.arange(np.ceil(5000 / subsampling_factor), dtype=torch.float32)
    yy, xx = torch.meshgrid(pix_range, pix_range, indexing="ij")
    return subsampling_factor * (torch.stack([xx, yy]) + 0.5)


def to_homogeneous(input_tensor, dim=1):
    """
    Converts tensor to homogeneous coordinates by adding ones to the specified dimension
    """
    ones = torch.ones_like(input_tensor.select(dim, 0).unsqueeze(dim))
    output = torch.cat([input_tensor, ones], dim=dim)
    return output


def read_nvm_file(file_name, return_rgb=False):
    with open(file_name) as file:
        lines = [line.rstrip() for line in file]
    nb_cameras = int(lines[2])
    image2info = {}
    image2pose = {}
    image2name = {}
    for i in range(nb_cameras):
        cam_info = lines[3 + i]
        img_name, info = cam_info.split("\t")
        focal, qw, qx, qy, qz, tx, ty, tz, radial, _ = map(float, info.split(" "))
        image2name[i] = img_name
        image2info[i] = [focal, radial]
        image2pose[i] = [qw, qx, qy, qz, tx, ty, tz]
    nb_points = int(lines[4 + nb_cameras])
    image2points = {}
    image2uvs = {}
    xyz_arr = np.zeros((nb_points, 3), np.float64)
    rgb_arr = np.zeros((nb_points, 3), np.uint8)
    for j in range(nb_points):
        point_info = lines[5 + nb_cameras + j].split(" ")
        x, y, z, r, g, b, nb_features = map(float, point_info[:7])
        xyz_arr[j] = [x, y, z]
        rgb_arr[j] = [r, g, b]
        features_info = point_info[7:]
        nb_features = int(nb_features)
        for k in range(nb_features):
            image_id, feature_id, u, v = features_info[k * 4 : (k + 1) * 4]
            image_id, feature_id = map(int, [image_id, feature_id])
            u, v = map(float, [u, v])
            image2points.setdefault(image_id, []).append(j)
            image2uvs.setdefault(image_id, []).append([u, v])

    if return_rgb:
        return xyz_arr, image2points, image2name, rgb_arr
    else:
        return xyz_arr, image2points, image2name


def return_pose_mat(pose_q, pose_t):
    pose_q = np.array([pose_q[1], pose_q[2], pose_q[3], pose_q[0]])
    pose_R = Rotation.from_quat(pose_q).as_matrix()

    pose_4x4 = np.identity(4)
    pose_4x4[0:3, 0:3] = pose_R
    pose_4x4[0:3, 3] = pose_t

    # convert world->cam to cam->world for evaluation
    pose_4x4_inv = np.linalg.inv(pose_4x4)
    return pose_4x4_inv
