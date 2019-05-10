from __future__ import division
import time
import torch
import numpy as np
from torch.autograd import Variable
import models.networks
from options.train_options import TrainOptions
import sys
from loaders import data_loader
from models import models
import random
import math

BATCH_SIZE = 1

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

eval_TUM_list_path = 'test_data/test_tum_hdf5_list.txt'

isTrain = False
eval_num_threads = 1
eval_data_loader = data_loader.CreateDataLoaderTUM(opt, eval_TUM_list_path,
                                                   isTrain, BATCH_SIZE,
                                                   eval_num_threads)
eval_dataset = eval_data_loader.load_data()
eval_data_size = len(eval_data_loader)
print('========================= TUM evaluation #images = %d =========' %
      eval_data_size)

model = models.create_model(opt, False)


def evaluation_tum(model, dataset, dataset_size, global_step):
  total_si_error = 0.0
  total_si_human_full_error = 0.0
  total_si_env_error = 0.0
  total_si_human_intra_error = 0.0
  total_si_human_inter_error = 0.0

  total_rel = 0.0
  total_rmse = 0.0

  count = 0.0
  save_img = False

  print(
      '============================= TUM Validation ============================'
  )
  print('dataset_size ', dataset_size)
  model.switch_to_eval()

  for i, data in enumerate(dataset):
    print(i)
    stacked_img = data[0]
    targets = data[1]

    if save_img:
      save_path = '/home/zhengqili/filestore_fast/Mannequin_depth_exp/comparisons/ours_ic_6_augmented/'
      print('save_path %s' % save_path)

      model.eval_save_tum_img(stacked_img, targets, save_path)
    else:
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


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
best_epoch = 0
global_step = 0

print(
    '=================================  BEGIN TUM VALIDATION ====================================='
)

print('TESTING ON TUM')
evaluation_tum(model, eval_dataset, eval_data_size, global_step)
