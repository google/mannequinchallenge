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

#from aligned_data_loader import * #AlignedDataLoader, TestDataLoader, TUMDataLoader
from loaders import aligned_data_loader

def CreateDataLoader(opt, img_dir, list_path, is_train, _batch_size, num_threads):
    data_loader = aligned_data_loader.AlignedDataLoader(opt, img_dir, list_path, is_train, _batch_size, num_threads) #CustomDatasetDataLoader()
    return data_loader

def CreateTestDataLoader(list_path, _batch_size):
    data_loader = aligned_data_loader.TestDataLoader(list_path, _batch_size) #CustomDatasetDataLoader()
    return data_loader

def CreateDAVISDataLoader(list_path, _batch_size):
    data_loader = aligned_data_loader.DAVISDataLoader(list_path, _batch_size) #CustomDatasetDataLoader()
    return data_loader

def CreateDataLoaderTUM(opt, img_dir, list_path, is_train, _batch_size, num_threads):
    tum_data_loader = aligned_data_loader.TUMDataLoader(opt, img_dir, list_path, is_train, _batch_size, num_threads) #CustomDatasetDataLoader()
    return tum_data_loader
