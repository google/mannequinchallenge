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

import argparse
import os
from util import util


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument(
            '--input', type=str, required=True, 
            choices=['single_view', 'two_view', 'two_view_k'],
            help='type of input. One of "single_view", "two_view" (no human keypoints),'
            '"two_view_k" (with human keypoints)')
        self.parser.add_argument('--simple_keypoints', type=int, default=0,
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--mode', type=str, default='Ours_Bilinear',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--human_data_term', type=int, default=0,
                                 help='which epoch to load? set to latest to use latest cached model')

        self.parser.add_argument(
            '--batchSize', type=int, default=8, help='input batch size')
        self.parser.add_argument(
            '--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument(
            '--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument(
            '--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument(
            '--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument(
            '--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--which_model_netG', type=str,
                                 default='unet_256', help='selects model to use for netG')
        self.parser.add_argument(
            '--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2')
        self.parser.add_argument('--name', type=str, default='test_local',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--model', type=str, default='pix2pix',
                                 help='chooses which model to use. cycle_gan, one_direction_test, pix2pix, ...')
        self.parser.add_argument(
            '--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str,
                                 default='./checkpoints/', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument(
            '--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument(
            '--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--identity', type=float, default=0.0,
                                 help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
        self.parser.add_argument(
            '--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float(
            "inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
