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

import random
import numpy as np
import torch.utils.data
from loaders import base_data_loader
#from loaders.image_folder import *
from loaders import image_folder
from builtins import object
import sys
import h5py

class AlignedDataLoader(base_data_loader.BaseDataLoader):
    def __init__(self, opt, img_dir, list_path, is_train, _batch_size, num_threads):
        dataset = image_folder.ImageFolder(opt=opt, img_dir=img_dir, 
            list_path =list_path, is_train=is_train)
        self.data_loader = torch.utils.data.DataLoader(dataset, 
            batch_size=_batch_size, shuffle=is_train, num_workers=int(num_threads))
        self.dataset = dataset

    def load_data(self):
        return self.data_loader

    def name(self):
        return 'AlignedDataLoader'
    
    def __len__(self):
        return len(self.dataset)


class TestDataLoader(base_data_loader.BaseDataLoader):
    def __init__(self, list_path, _batch_size):
        dataset = image_folder.TestImageFolder(list_path =list_path)
        self.data_loader = torch.utils.data.DataLoader(dataset, 
            batch_size=_batch_size, shuffle=False, num_workers=int(1))
        self.dataset = dataset

    def load_data(self):
        return self.data_loader

    def name(self):
        return 'TestDataLoader'
    
    def __len__(self):
        return len(self.dataset)


class DAVISDataLoader(base_data_loader.BaseDataLoader):
    def __init__(self, list_path, _batch_size):
        dataset = image_folder.DAVISImageFolder(list_path =list_path)
        self.data_loader = torch.utils.data.DataLoader(dataset, 
            batch_size=_batch_size, shuffle=False, num_workers=int(1))
        self.dataset = dataset

    def load_data(self):
        return self.data_loader

    def name(self):
        return 'TestDataLoader'
    
    def __len__(self):
        return len(self.dataset)


class TUMDataLoader(base_data_loader.BaseDataLoader):
    def __init__(self, opt, img_dir, list_path, is_train, _batch_size, num_threads):
        dataset = image_folder.TUMImageFolder(opt=opt, list_path =list_path)
        self.data_loader = torch.utils.data.DataLoader(dataset, 
            batch_size=_batch_size, shuffle=False, num_workers=int(num_threads))
        self.dataset = dataset

    def load_data(self):
        return self.data_loader

    def name(self):
        return 'TUMDataLoader'
    
    def __len__(self):
        return len(self.dataset)
