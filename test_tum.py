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

import torch
from options.train_options import TrainOptions
from loaders import aligned_data_loader
from models import pix2pix_model

BATCH_SIZE = 1

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

eval_TUM_list_path = 'test_data/test_tum_hdf5_list.txt'

isTrain = False
eval_num_threads = 1
eval_data_loader = aligned_data_loader.TUMDataLoader(opt, eval_TUM_list_path,
                                                     isTrain, BATCH_SIZE,
                                                     eval_num_threads)
dataset = eval_data_loader.load_data()
data_size = len(eval_data_loader)
print('========================= TUM evaluation #images = %d =========' %
      data_size)

model = pix2pix_model.Pix2PixModel(opt, False)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
best_epoch = 0
global_step = 0

print(
    '=================================  BEGIN TUM VALIDATION ====================================='
)

print('TESTING ON TUM')
total_si_error = 0.0
total_si_human_full_error = 0.0
total_si_env_error = 0.0
total_si_human_intra_error = 0.0
total_si_human_inter_error = 0.0

total_rel = 0.0
total_rmse = 0.0

count = 0.0

print(
    '============================= TUM Validation ============================'
)
model.switch_to_eval()

for i, data in enumerate(dataset):
    print(i)
    stacked_img = data[0]
    targets = data[1]

    sc_inv_errors, rel_error, RMSE_error = model.evaluate_tum_error(
        stacked_img, targets, global_step, False)

    count += stacked_img.size(0)
    total_si_error += sc_inv_errors[0]
    total_si_human_full_error += sc_inv_errors[1]
    total_si_env_error += sc_inv_errors[2]
    total_si_human_intra_error += sc_inv_errors[3]
    total_si_human_inter_error += sc_inv_errors[4]

    total_rel += rel_error
    total_rmse += RMSE_error

    sc_inv_rmse = float(total_si_error) / float(count)
    sc_inv_human_rmse = float(total_si_human_full_error) / float(count)
    sc_inv_env_rmse = float(total_si_env_error) / float(count)
    sc_inv_intra_rmse = float(
        total_si_human_intra_error) / float(count)
    sc_inv_inter_rmse = float(
        total_si_human_inter_error) / float(count)

    rel_avg = float(total_rel) / float(count)
    rmse_avg = float(total_rmse) / float(count)

    print('============== Sc-inv full RMSE: %f' % sc_inv_rmse)
    print('============== Sc-inv Human Full RMSE: %f' %
          sc_inv_human_rmse)
    print('============== Sc-inv Human Intra RMSE: %f' %
          sc_inv_intra_rmse)
    print('============== Sc-inv Human Inter RMSE: %f' %
          sc_inv_inter_rmse)
    print('============== Sc-inv Env RMSE: %f' % sc_inv_env_rmse)
    print('============== rel_avg: %f' % rel_avg)
    print('============== rmse_avg: %f' % rmse_avg)

sc_inv_rmse = float(total_si_error) / float(count)
sc_inv_human_rmse = float(total_si_human_full_error) / float(count)
sc_inv_env_rmse = float(total_si_env_error) / float(count)
sc_inv_intra_rmse = float(total_si_human_intra_error) / float(count)
sc_inv_inter_rmse = float(total_si_human_inter_error) / float(count)

rel_avg = float(total_rel) / float(count)
rmse_avg = float(total_rmse) / float(count)

print('============== Sc-inv full RMSE: %f' % sc_inv_rmse)
print('============== Sc-inv Human Full RMSE: %f' % sc_inv_human_rmse)
print('============== Sc-inv Human Intra RMSE: %f' % sc_inv_intra_rmse)
print('============== Sc-inv Human Inter RMSE: %f' % sc_inv_inter_rmse)
print('============== Sc-inv Env RMSE: %f' % sc_inv_env_rmse)
print('============== rel_avg: %f' % rel_avg)
print('============== rmse_avg: %f' % rmse_avg)
