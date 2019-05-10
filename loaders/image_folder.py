'''
Copyright 2019 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

################################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
################################################################################
import h5py
import torch.utils.data as data
import pickle
import PIL
import numpy as np
import torch
from PIL import Image
import os
import math
import random
import os.path
import sys
import traceback
from skimage import transform
from skimage.io import imread

from skimage.morphology import erosion
from skimage.morphology import disk
from skimage.morphology import dilation

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


def make_color_wheel():
    """Generate color wheel according Middlebury color code.

    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - \
        np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC,
               2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - \
        np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM,
               0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - \
        np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


def compute_color(u, v):
    """compute optical flow color map.

    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img


def flow_to_image(flow, display=False):
    """Convert flow into middlebury color code image.

    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    UNKNOWN_FLOW_THRESH = 100
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    # sqrt_rad = u**2 + v**2
    rad = np.sqrt(u**2 + v**2)

    maxrad = max(-1, np.max(rad))

    if display:
        print('max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f' % (
            maxrad, minu, maxu, minv, maxv))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


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


class ImageFolder(data.Dataset):

    def __init__(self, opt, img_dir, list_path, is_train):
        img_list = make_dataset(list_path)
        if len(img_list) == 0:
            raise(RuntimeError('Found 0 images in: ' + root + '\n'
                               'Supported image extensions are: ' + ','.join(IMG_EXTENSIONS)))
        self.img_dir = img_dir
        self.list_path = list_path
        self.img_list = img_list
        self.opt = opt
        self.input_width = 512
        self.input_height = 288

        self.is_train = is_train
        self.resized_w = 532  # 532 #542
        self.resized_h = 299  # 299 #305

        self.div_flow = 10.0
        xx = range(0, self.resized_w)
        yy = range(0, self.resized_h)  # , self.resized_h)
        xv, yv = np.meshgrid(xx, yy)
        self.p_1 = np.float32(np.stack((xv, yv), axis=-1))
        self.p_1_h = np.reshape(self.p_1, (-1, 2))
        self.p_1_h = np.concatenate(
            (self.p_1_h, np.ones((self.p_1_h.shape[0], 1))), axis=-1).T

    def load_imgs(self, img_path):
        img = imread(img_path)
        return img

    def load_hdf5(self, hdf5_path):

        lr_t = 1.0
        alpha_t = 1.0
        rel_diff_t = 0.2
        # print('hdf5_path', hdf5_path)

        with h5py.File(hdf5_path, 'r') as hdf5_file_read:
            img = hdf5_file_read.get('/gt/img')
            img = np.float32(np.array(img))/255.0

            img_2 = hdf5_file_read.get('/gt/img_2')
            img_2 = np.float32(np.array(img_2))/255.0

            depth_gt = hdf5_file_read.get('/gt/mvs_depth')
            depth_gt = np.float32(np.array(depth_gt))
            original_gt_mask = np.float32(depth_gt > 1e-8)

            full_flow = hdf5_file_read.get('/gt/flow')
            full_flow = np.float32(full_flow)

            lr_error = hdf5_file_read.get('/gt/lr_error')
            lr_error = np.float32(np.array(lr_error))
            lr_prob = np.maximum(0.0, 1.0 - lr_error**2)

            env_mask = hdf5_file_read.get('/gt/human_mask')
            env_mask = np.float32(np.array(env_mask))
            human_mask = 1.0 - env_mask

            view_angle = hdf5_file_read.get('/gt/angle_prior')
            view_angle = np.float32(np.array(view_angle))

            # mvs_flow = hdf5_file_read.get('/gt/mvs_flow');
            # mvs_flow = np.float32(np.array(mvs_flow))

            pp_depth_gt = hdf5_file_read.get('/gt/pp_depth')
            pp_depth_gt = np.float32(np.array(pp_depth_gt))

            pp_depth_gt_original = pp_depth_gt.copy()

            angle_prob = 1.0 - \
                ((np.minimum(alpha_t, view_angle) - alpha_t)**2)/(alpha_t**2)

            rel_error = np.abs(depth_gt - pp_depth_gt) / \
                (pp_depth_gt + depth_gt + 1e-8)

            flow_check_mask = np.float32(
                rel_error > rel_diff_t) * np.float32(lr_error < 2.0) * np.float32(angle_prob > 0.5)
            flow_check_mask = 1.0 - flow_check_mask * original_gt_mask

            gt_mask = original_gt_mask * flow_check_mask

            # random mask for pp depth
            if self.is_train:
                rd_arr = [0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.5, 2.0, 3.0, 4.0]
                rd_lr_idx = np.random.randint(0, len(rd_arr))
                lr_threshold = rd_arr[rd_lr_idx]
            else:
                lr_threshold = 1.0

            lr_mask = np.float32(lr_error < lr_threshold)
            angle_mask = np.float32(angle_prob > pa_t)

            pp_mask = angle_mask * lr_mask * env_mask
            pp_depth_gt = pp_depth_gt * angle_mask * lr_mask

            # load keypoints
            keypoints_img = hdf5_file_read.get('/gt/keypoints_img')
            keypoints_img = np.int32(np.array(keypoints_img))

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

            geometric_e_distance = compute_epipolar_distance(
                T_21, K, p_1_h, p_2_h)
            geometric_e_distance = np.abs(geometric_e_distance)
            geometric_e_distance = np.reshape(
                geometric_e_distance, (img.shape[0], img.shape[1]))
            # epipolar_mask = np.float32(geometric_e_distance < e_t)

            e_likelihood = np.maximum(
                0.0, 1.0 - (geometric_e_distance**2)/(e_t**2))

            if self.is_train:
                keypoints_mask = np.float32(keypoints_img > 1e-8)
                joint_mask = np.float32(
                    angle_prob > 0.5) * np.float32(lr_error < lr_t) * np.float32(e_likelihood > 1e-8)
                flow_mask = keypoints_mask * \
                    (1.0 - original_gt_mask) * joint_mask * \
                    np.float32(pp_depth_gt > 1e-8)
                depth_gt = pp_depth_gt * flow_mask + depth_gt * gt_mask
                gt_mask = np.float32((gt_mask + flow_mask) > 1e-8)

            pp_depth = pp_depth_gt * pp_mask
            input_confidence = lr_prob * angle_prob * pp_mask * e_likelihood

            # avoid numerical issue (log negative number )
            depth_gt[depth_gt < 1e-8] = 1.0

        return {'img': img,
                'img_2': img_2,
                'depth_gt': depth_gt,
                'gt_mask': gt_mask,
                'keypoints_img': keypoints_img,
                'full_flow': full_flow,
                'env_mask': env_mask,
                'pp_depth': pp_depth,
                'pp_depth_gt': pp_depth_gt_original,
                'input_confidence': input_confidence,
                'K': K,
                'T_1_G': T_1_G}

    def da_crop_img(self, train_data):
        resized_height, resized_width = train_data['depth_gt'].shape[0], train_data['depth_gt'].shape[1]

        if self.is_train:
            start_y = random.randint(0, resized_height - self.input_height)
            start_x = random.randint(0, resized_width - self.input_width)
        else:
            start_y = int((resized_height - self.input_height)/2.0)
            start_x = int((resized_width - self.input_width)/2.0)

            train_data['K'][0, 2] = train_data['K'][0, 2] - start_x
            train_data['K'][1, 2] = train_data['K'][1, 2] - start_y

        train_data['img'] = train_data['img'][start_y:start_y +
                                              self.input_height, start_x:start_x+self.input_width, :]
        train_data['img_2'] = train_data['img_2'][start_y:start_y +
                                                  self.input_height, start_x:start_x+self.input_width, :]

        train_data['depth_gt'] = train_data['depth_gt'][start_y:start_y +
                                                        self.input_height, start_x:start_x+self.input_width]
        train_data['pp_depth'] = train_data['pp_depth'][start_y:start_y +
                                                        self.input_height, start_x:start_x+self.input_width]
        train_data['pp_depth_gt'] = train_data['pp_depth_gt'][start_y:start_y +
                                                              self.input_height, start_x:start_x+self.input_width]
        train_data['keypoints_img'] = train_data['keypoints_img'][start_y:start_y +
                                                                  self.input_height, start_x:start_x+self.input_width]

        train_data['full_flow'] = train_data['full_flow'][start_y:start_y +
                                                          self.input_height, start_x:start_x+self.input_width, :]

        train_data['input_confidence'] = train_data['input_confidence'][start_y:start_y +
                                                                        self.input_height, start_x:start_x+self.input_width]
        train_data['env_mask'] = train_data['env_mask'][start_y:start_y +
                                                        self.input_height, start_x:start_x+self.input_width]
        train_data['gt_mask'] = train_data['gt_mask'][start_y:start_y +
                                                      self.input_height, start_x:start_x+self.input_width]

        return train_data

    def random_normaliztion(self, input_depth, input_mask, random_percentile):
        input_log_depth = np.log(input_depth)
        median_log_depth = 0.0
        if np.any(input_mask):
            median_log_depth = np.percentile(
                input_log_depth[input_mask], random_percentile)

        input_log_depth = (input_log_depth - median_log_depth) * \
            np.float32(input_mask)
        return input_log_depth

    def __getitem__(self, index):
        targets_1 = {}

        h5_path = self.img_list[index].rstrip()
        # h5_path = h5_path.replace('hdf5', 'new_hdf5')

        train_data = self.load_hdf5(h5_path)
        train_data = self.da_crop_img(train_data)

        img = train_data['img']
        img_2 = train_data['img_2']

        depth_gt = train_data['depth_gt']
        gt_mask = train_data['gt_mask']
        input_confidence = train_data['input_confidence']
        env_mask = train_data['env_mask']
        pp_depth = train_data['pp_depth']
        keypoints_img = train_data['keypoints_img']
        full_flow = train_data['full_flow']
        K = train_data['K']
        T_1_G = train_data['T_1_G']
        pp_depth_gt = train_data['pp_depth_gt']

        prob_flip = random.random()

        if prob_flip > 0.5 and self.is_train:
            img = np.fliplr(img)
            depth_gt = np.fliplr(depth_gt)
            gt_mask = np.fliplr(gt_mask)
            env_mask = np.fliplr(env_mask)
            input_confidence = np.fliplr(input_confidence)
            pp_depth = np.fliplr(pp_depth)
            pp_depth_gt = np.fliplr(pp_depth_gt)

        # flip keypoints needs more careful
        if self.opt.simple_keypoints == 1:
            # print('simple_keypoints', 1)
            keypoints_img = np.vectorize(
                keypoints_simple_dict.__getitem__)(keypoints_img)
            keypoints_img = np.float32(keypoints_img)/keypoints_simple_factor

            if prob_flip > 0.5 and self.is_train:
                keypoints_img = np.fliplr(keypoints_img)

        elif self.opt.simple_keypoints == 2:
            # print('simple_keypoints', 2)
            keypoints_img = np.float32(keypoints_img)/keypoints_factor

            if prob_flip > 0.5 and self.is_train:
                keypoints_img = np.fliplr(keypoints_img)
        else:
            if prob_flip > 0.5 and self.is_train:
                keypoints_img = np.vectorize(
                    keypoints_flip_dict.__getitem__)(keypoints_img)
                keypoints_img = np.fliplr(keypoints_img)

            keypoints_img = np.float32(keypoints_img)/keypoints_factor

        # mask human region
        input_depth = pp_depth  # * env_mask
        # convert to log depth
        input_mask = input_depth > 1e-8
        input_depth[input_depth < 1e-8] = 1.0

        if self.is_train:
            random_percentile = np.random.randint(40, 60)
            input_log_depth = self.random_normaliztion(
                input_depth, input_mask, random_percentile)
        else:
            input_log_depth = self.random_normaliztion(
                input_depth, input_mask, 50)

        final_img = torch.from_numpy(np.ascontiguousarray(
            img).transpose(2, 0, 1)).contiguous().float()

        targets_1['input_confidence'] = torch.from_numpy(
            np.ascontiguousarray(input_confidence)).contiguous().float()
        targets_1['input_log_depth'] = torch.from_numpy(
            np.ascontiguousarray(input_log_depth)).contiguous().float()

        targets_1['depth_gt'] = torch.from_numpy(
            np.ascontiguousarray(depth_gt)).contiguous().float()
        targets_1['gt_mask'] = torch.from_numpy(
            np.ascontiguousarray(gt_mask)).contiguous().float()
        targets_1['pp_depth_gt'] = torch.from_numpy(
            np.ascontiguousarray(pp_depth_gt)).contiguous().float()
        targets_1['keypoints_img'] = torch.from_numpy(
            np.ascontiguousarray(keypoints_img)).contiguous().float()
        targets_1['input_depth_final'] = torch.from_numpy(
            np.ascontiguousarray(input_depth)).contiguous().float()
        targets_1['full_flow'] = torch.from_numpy(
            np.ascontiguousarray(full_flow)).contiguous().float()
        targets_1['img_2'] = torch.from_numpy(
            np.ascontiguousarray(img_2)).contiguous().float()

        targets_1['env_mask'] = torch.from_numpy(
            np.ascontiguousarray(env_mask)).contiguous().float()
        targets_1['K'] = torch.from_numpy(
            np.ascontiguousarray(K)).contiguous().float()
        targets_1['T_1_G'] = torch.from_numpy(
            np.ascontiguousarray(T_1_G)).contiguous().float()

        targets_1['img_1_path'] = h5_path

        return final_img, targets_1

    def __len__(self):
        return len(self.img_list)


class TestImageFolder(data.Dataset):

    def __init__(self, list_path):
        img_list = make_dataset(list_path)
        if len(img_list) == 0:
            raise(RuntimeError('Found 0 images in: ' + root + '\n'
                               'Supported image extensions are: ' + ','.join(IMG_EXTENSIONS)))
        self.list_path = list_path
        self.img_list = img_list

        self.resized_height = 288
        self.resized_width = 512

        self.input_height = 256
        self.input_width = 512

        self.input_mode = 0  # 0:pp

        xx = range(0, self.resized_width)
        yy = range(0, self.resized_height)  # , self.resized_h)
        xv, yv = np.meshgrid(xx, yy)
        self.p_1 = np.float32(np.stack((xv, yv), axis=-1))
        self.p_1_h = np.reshape(self.p_1, (-1, 2))
        self.p_1_h = np.concatenate(
            (self.p_1_h, np.ones((self.p_1_h.shape[0], 1))), axis=-1).T

    def load_imgs(self, img_path):
        img = imread(img_path)
        return img

    def load_hdf5(self, hdf5_path):

        lr_threshold = 1.0

        # print('hdf5_path ', hdf5_path)

        with h5py.File(hdf5_path, 'r') as hdf5_file_read:
            img = hdf5_file_read.get('/gt/im1')
            img = np.float32(np.array(img))

            gray_scale_img = np.mean(img, axis=-1)

            lr_error = hdf5_file_read.get('/gt/lr_error')
            lr_error = np.float32(np.array(lr_error))
            lr_prob = np.maximum(0.0, 1.0 - (lr_error/lr_threshold)**2)

            human_mask = hdf5_file_read.get('/gt/human_mask')
            env_mask = 1.0 - np.float32(np.array(human_mask))

            view_angle = hdf5_file_read.get('/gt/angle_prior')
            view_angle = np.float32(np.array(view_angle))

            pp_depth = hdf5_file_read.get('/gt/pp_depth')
            pp_depth = np.float32(np.array(pp_depth))
            angle_prob = 1.0 - \
                ((np.minimum(alpha_t, view_angle) - alpha_t)**2)/(alpha_t**2)

            full_flow = hdf5_file_read.get('/gt/flow')
            full_flow = np.float32(full_flow)

            lr_mask = np.float32(lr_error < 1.0)
            angle_mask = np.float32(angle_prob > 0.25)

            pp_mask = angle_mask * lr_mask * env_mask
            pp_mask = erosion(pp_mask > 0.1, disk(1))

            pp_depth = pp_depth * pp_mask

            mvs_depth = hdf5_file_read.get('/gt/mvs_depth')
            mvs_depth = np.float32(np.array(mvs_depth))
            original_mvs_depth = mvs_depth.copy() * env_mask
            mvs_depth = mvs_depth * env_mask

            gt_mask = np.float32(mvs_depth > 1e-8)

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

            geometric_e_distance = compute_epipolar_distance(
                T_21, K, p_1_h, p_2_h)
            geometric_e_distance = np.abs(geometric_e_distance)
            geometric_e_distance = np.reshape(
                geometric_e_distance, (img.shape[0], img.shape[1]))
            epipolar_mask = np.float32(geometric_e_distance < 5.0)

            e_likelihood = np.maximum(
                0.0, 1.0 - (geometric_e_distance**2)/(e_t**2))

            # rel_error = np.abs(mvs_depth - pp_depth)/(pp_depth + mvs_depth + 1e-8)
            # flow_check_mask = np.float32(rel_error > 0.2) * np.float32(angle_prob > 0.5) * lr_mask * gt_mask * np.float32(pp_depth > 1e-8)
            # flow_check_mask = 1.0 - flow_check_mask

            # sky_mask = np.float32(gray_scale_img < 0.95)
            pp_depth = pp_depth * epipolar_mask * pp_mask

            input_confidence = lr_prob * angle_prob * e_likelihood * pp_mask
            input_flow = full_flow * np.expand_dims(env_mask, axis=-1)

            keypoints_img = hdf5_file_read.get('/gt/keypoints_img')
            keypoints_img = np.float32(np.array(keypoints_img))/18.0

        return {'img': img,
                'env_mask': env_mask,
                'pp_depth': pp_depth,
                'gt_mask': gt_mask,
                'mvs_depth': mvs_depth,
                'full_flow': full_flow,
                'original_mvs_depth': original_mvs_depth,
                'input_confidence': input_confidence,
                'keypoints_img': keypoints_img,
                'K': K,
                'T_1_G': T_1_G}

    def __getitem__(self, index):
        targets_1 = {}

        h5_path = self.img_list[index].rstrip()
        train_data = self.load_hdf5(h5_path)

        img = train_data['img']
        input_confidence = train_data['input_confidence']
        env_mask = train_data['env_mask']
        pp_depth = train_data['pp_depth']
        mvs_depth = train_data['mvs_depth']
        original_mvs_depth = train_data['original_mvs_depth']
        keypoints_img = train_data['keypoints_img']

        gt_mask = train_data['gt_mask']
        full_flow = train_data['full_flow']
        K = train_data['K']
        T_1_G = train_data['T_1_G']

        # if you want p+p depth
        if self.input_mode == 0:
            # print('use pp')
            input_depth = pp_depth * env_mask
        else:
            print('SOMETHING WRONG')
            sys.exit()

        # convert to log depth
        original_pp_depth = input_depth + 0.0  # pp_depth * env_mask

        input_mask = input_depth > 1e-8
        input_depth[input_depth < 1e-8] = 1.0
        input_log_depth = np.log(input_depth)
        median_log_depth = 0.0

        if np.any(input_mask):
            median_log_depth = np.median(input_log_depth[input_mask])

        input_log_depth = (input_log_depth - median_log_depth) * \
            np.float32(input_mask)

        full_flow_rgb = flow_to_image(full_flow)

        final_img = torch.from_numpy(np.ascontiguousarray(
            img).transpose(2, 0, 1)).contiguous().float()
        targets_1['input_confidence'] = torch.from_numpy(
            np.ascontiguousarray(input_confidence)).contiguous().float()
        targets_1['input_log_depth'] = torch.from_numpy(
            np.ascontiguousarray(input_log_depth)).contiguous().float()
        targets_1['env_mask'] = torch.from_numpy(
            np.ascontiguousarray(env_mask)).contiguous().float()
        targets_1['input_depth'] = torch.from_numpy(
            np.ascontiguousarray(original_pp_depth)).contiguous().float()
        targets_1['mvs_depth'] = torch.from_numpy(
            np.ascontiguousarray(mvs_depth)).contiguous().float()
        targets_1['original_mvs_depth'] = torch.from_numpy(
            np.ascontiguousarray(original_mvs_depth)).contiguous().float()

        targets_1['full_flow'] = torch.from_numpy(
            np.ascontiguousarray(full_flow)).contiguous().float()
        targets_1['keypoints_img'] = torch.from_numpy(
            np.ascontiguousarray(keypoints_img)).contiguous().float()

        targets_1['K'] = torch.from_numpy(
            np.ascontiguousarray(K)).contiguous().float()
        targets_1['T_1_G'] = torch.from_numpy(
            np.ascontiguousarray(T_1_G)).contiguous().float()
        targets_1['img_1_path'] = h5_path

        return final_img, targets_1

    def __len__(self):
        return len(self.img_list)


class TUMImageFolder(data.Dataset):

    def __init__(self, opt, list_path):
        img_list = make_dataset(list_path)
        if len(img_list) == 0:
            raise(RuntimeError('Found 0 images in: ' + root + '\n'
                               'Supported image extensions are: ' + ','.join(IMG_EXTENSIONS)))
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
            # remove moving objects
            # epipolar_mask = np.float32(geometric_e_distance < e_t**2)

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
        # full_flow = train_data['full_flow']
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
            raise(RuntimeError('Found 0 images in: ' + root + '\n'
                               'Supported image extensions are: ' + ','.join(IMG_EXTENSIONS)))
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
