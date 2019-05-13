# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import h5py
import torch.utils.data as data
import numpy as np
import torch
import os
import os.path
from skimage import transform
from skimage.io import imread


keypoints_simple_dict = {0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4,
                         8: 5, 9: 5, 10: 6, 11: 6, 12: 7, 13: 7, 14: 8, 15: 8, 16: 9, 17: 9}
keypoints_flip_dict = {0: 0, 1: 1, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6, 8: 9,
                       9: 8, 10: 11, 11: 10, 12: 13, 13: 12, 14: 15, 15: 14, 16: 17, 17: 16}
patch_simple_dict = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 9: 6, 8: 7, 10: 7, 11: 8, 13: 8,
                     12: 9, 14: 9, 15: 10, 17: 10, 16: 11, 18: 11, 19: 12, 21: 12, 20: 13, 22: 13, 23: 14, 24: 14}

lr_t = 1.0
alpha_t = 1.0
e_t = 2.0
rel_diff_t = 0.2
keypoints_simple_factor = 10.0
keypoints_factor = 18.0
pa_t = 0.25


def make_dataset(list_name):
    text_file = open(list_name, 'r')
    images_list = text_file.readlines()
    text_file.close()
    images_list = [os.path.join(os.getcwd(), i) for i in images_list]
    return images_list


def read_array(path):
    with open(path, 'rb') as fid:
        width, height, channels = np.genfromtxt(fid, delimiter='&', max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b'&':
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order='F')
    return np.transpose(array, (1, 0, 2)).squeeze()


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def compute_epipolar_distance(T_21, K, p_1, p_2):
    R_21 = T_21[:3, :3]
    t_21 = T_21[:3, 3]

    E_mat = np.dot(skew(t_21), R_21)
    # compute bearing vector
    inv_K = np.linalg.inv(K)

    F_mat = np.dot(np.dot(inv_K.T, E_mat), inv_K)

    l_2 = np.dot(F_mat, p_1)
    algebric_e_distance = np.sum(p_2 * l_2, axis=0)
    n_term = np.sqrt(l_2[0, :]**2 + l_2[1, :]**2) + 1e-8
    geometric_e_distance = algebric_e_distance/n_term

    return geometric_e_distance


class TUMImageFolder(data.Dataset):

    def __init__(self, opt, list_path):
        img_list = make_dataset(list_path)
        if len(img_list) == 0:
            raise(RuntimeError('Found 0 images in: ' + list_path))
        self.list_path = list_path
        self.img_list = img_list
        self.opt = opt
        self.resized_height = 384
        self.resized_width = 512

        self.input_height = 384
        self.input_width = 512

        xx = range(0, self.resized_width)
        yy = range(0, self.resized_height)  # , self.resized_h)
        xv, yv = np.meshgrid(xx, yy)
        self.p_1 = np.float32(np.stack((xv, yv), axis=-1))
        self.p_1_h = np.reshape(self.p_1, (-1, 2))
        self.p_1_h = np.concatenate(
            (self.p_1_h, np.ones((self.p_1_h.shape[0], 1))), axis=-1).T

    def load_tum_hdf5(self, hdf5_path):

        with h5py.File(hdf5_path, 'r') as hdf5_file_read:
            img = hdf5_file_read.get('/gt/img_1')
            img = np.float32(np.array(img))

            depth_gt = hdf5_file_read.get('/gt/gt_depth')
            depth_gt = np.float32(np.array(depth_gt))
            gt_mask = np.float32(depth_gt > 1e-8)

            lr_error = hdf5_file_read.get('/gt/lr_error')
            lr_error = np.float32(np.array(lr_error))
            lr_prob = np.maximum(0.0, 1.0 - lr_error**2/(lr_t**2))

            human_mask = hdf5_file_read.get('/gt/human_mask')
            env_mask = 1.0 - np.float32(np.array(human_mask))

            view_angle = hdf5_file_read.get('/gt/angle_prior')
            view_angle = np.float32(np.array(view_angle))

            pp_depth = hdf5_file_read.get('/gt/pp_depth')
            pp_depth = np.float32(np.array(pp_depth))

            angle_prob = 1.0 - \
                ((np.minimum(alpha_t, view_angle) - alpha_t)**2)/(alpha_t**2)

            full_flow = hdf5_file_read.get('/gt/flow')
            full_flow = np.float32(np.array(full_flow))

            T_1_G = hdf5_file_read.get('/gt/T_1_G')
            T_1_G = np.float32(np.array(T_1_G))

            T_2_G = hdf5_file_read.get('/gt/T_2_G')
            T_2_G = np.float32(np.array(T_2_G))

            K = hdf5_file_read.get('/gt/intrinsic')
            K = np.float32(np.array(K))

            T_21 = np.dot(T_2_G, np.linalg.inv(T_1_G))
            p_2 = self.p_1 + full_flow
            p_1_h = self.p_1_h
            p_2_h = np.reshape(p_2, (-1, 2))
            p_2_h = np.concatenate(
                (p_2_h, np.ones((p_2_h.shape[0], 1))), axis=-1).T

            # compute epipolar constraints
            geometric_e_distance = compute_epipolar_distance(
                T_21, K, p_1_h, p_2_h)
            geometric_e_distance = np.abs(geometric_e_distance)
            geometric_e_distance = np.reshape(
                geometric_e_distance, (img.shape[0], img.shape[1]))
            e_likelihood = np.maximum(
                0.0, 1.0 - (geometric_e_distance**2)/(e_t**2))

            lr_mask = lr_error < lr_t
            angle_mask = angle_prob > pa_t
            pp_mask = angle_mask * lr_mask * env_mask

            pp_depth = pp_depth * pp_mask
            input_confidence = lr_prob * angle_prob * pp_mask * e_likelihood

            depth_gt[depth_gt < 1e-8] = 1.0

            keypoints_img = hdf5_file_read.get('/gt/keypoints_img')
            keypoints_img = np.int32(np.array(keypoints_img))

        return {'img': img,
                'depth_gt': depth_gt,
                'gt_mask': gt_mask,
                'keypoints_img': keypoints_img,
                'env_mask': env_mask,
                'pp_depth': pp_depth,
                'input_confidence': input_confidence}

    def __getitem__(self, index):
        targets_1 = {}

        h5_path = self.img_list[index].rstrip()

        train_data = self.load_tum_hdf5(h5_path)

        img = train_data['img']
        depth_gt = train_data['depth_gt']
        gt_mask = train_data['gt_mask']
        input_confidence = train_data['input_confidence']
        env_mask = train_data['env_mask']
        pp_depth = train_data['pp_depth']
        keypoints_img = train_data['keypoints_img']

        if self.opt.simple_keypoints == 1:
            keypoints_img = np.vectorize(
                keypoints_simple_dict.__getitem__)(keypoints_img)
            keypoints_img = np.float32(keypoints_img)/keypoints_simple_factor
        elif self.opt.simple_keypoints == 2:
            keypoints_img = np.float32(keypoints_img)/keypoints_factor
        else:
            keypoints_img = np.float32(keypoints_img)/keypoints_factor

        # mask human region
        input_depth = pp_depth * env_mask
        # convert to log depth
        input_mask = input_depth > 1e-8
        input_depth[input_depth < 1e-8] = 1.0
        input_log_depth = np.log(input_depth)
        median_log_depth = 0.0
        if np.any(input_mask):
            median_log_depth = np.median(input_log_depth[input_mask])

        input_log_depth = (input_log_depth - median_log_depth) * \
            np.float32(input_mask)

        targets_1['img_1_path'] = h5_path
        final_img = torch.from_numpy(
            img.transpose(2, 0, 1)).contiguous().float()
        targets_1['depth_gt'] = torch.from_numpy(
            np.ascontiguousarray(depth_gt)).contiguous().float()
        targets_1['gt_mask'] = torch.from_numpy(
            np.ascontiguousarray(gt_mask)).contiguous().float()
        targets_1['env_mask'] = torch.from_numpy(
            np.ascontiguousarray(env_mask)).contiguous().float()
        targets_1['input_confidence'] = torch.from_numpy(
            np.ascontiguousarray(input_confidence)).contiguous().float()
        targets_1['input_log_depth'] = torch.from_numpy(
            np.ascontiguousarray(input_log_depth)).contiguous().float()
        targets_1['keypoints_img'] = torch.from_numpy(
            np.ascontiguousarray(keypoints_img)).contiguous().float()

        return final_img, targets_1

    def __len__(self):
        return len(self.img_list)


class DAVISImageFolder(data.Dataset):

    def __init__(self, list_path):
        img_list = make_dataset(list_path)
        if len(img_list) == 0:
            raise(RuntimeError('Found 0 images in: ' + list_path))
        self.list_path = list_path
        self.img_list = img_list

        self.resized_height = 288
        self.resized_width = 512

        self.use_pp = True

    def load_imgs(self, img_path):
        img = imread(img_path)
        img = np.float32(img)/255.0
        img = transform.resize(img, (self.resized_height, self.resized_width))

        return img

    def __getitem__(self, index):
        targets_1 = {}

        h5_path = self.img_list[index].rstrip()
        img = self.load_imgs(h5_path)

        final_img = torch.from_numpy(np.ascontiguousarray(
            img).transpose(2, 0, 1)).contiguous().float()

        targets_1['img_1_path'] = h5_path

        return final_img, targets_1

    def __len__(self):
        return len(self.img_list)
