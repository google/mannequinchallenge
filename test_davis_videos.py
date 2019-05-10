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

from __future__ import division
import time
import torch
import numpy as np
from torch.autograd import Variable
import models.networks
from options.train_options import TrainOptions
import sys
from loaders import data_loader
from models.models import create_model
import random
import math

BATCH_SIZE = 8

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

root = "/"
#img_dir = '/home/zhengqili/filestore/DAVIS/images/'
video_list = 'test_data/test_davis_video_list.txt'

isTrain = False
eval_num_threads = 2
video_data_loader = data_loader.CreateDAVISDataLoader(video_list, BATCH_SIZE)
video_dataset = video_data_loader.load_data()
video_data_size = len(video_data_loader)
print('========================= Video dataset #images = %d ========='%video_data_size)

model = create_model(opt, isTrain)
# model.switch_to_train()

def test_video(model, dataset, dataset_size):

    model.switch_to_eval()
    save_path = 'test_data/viz_predictions/'         
    print('save_path %s'%save_path)

    for i, data in enumerate(dataset):
        print(i)
        stacked_img = data[0]
        targets = data[1]

        model.run_and_save_DAVIS(stacked_img, targets, save_path)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
best_epoch =0
global_step = 0


print("=================================  BEGIN VALIDATION =====================================")

print('TESTING ON VIDEO')
test_video(model, video_dataset, video_data_size)
