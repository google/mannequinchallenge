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
