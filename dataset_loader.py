# dataset_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import math

class AccidentDataset(Dataset):
    def __init__(self, path, file_list=None):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Features directory '{path}' does not exist.")

        self.path = path
        if file_list is None:
            files = [f for f in os.listdir(path) if f.endswith('.npz')]
        else:
            files = list(file_list)

        self.files = sorted(files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.path, self.files[idx])
        with np.load(file_path, allow_pickle=True) as data:
            features = data['data']
            labels = data['labels']
            det = data['det']
            ids = data['ID']
        return torch.from_numpy(features).float(), torch.from_numpy(labels).float(), det, ids

def collate_fn(batch):
    """
    Custom collate function to handle batching of data from .npz files,
    as each item from the dataset is already a 'batch' from one file.
    This function simply unpacks the first (and only) item.
    """
    # Each 'batch' from DataLoader will be a list containing one tuple:
    # [(features_tensor, labels_tensor, det_array, ids_array)]
    features, labels, det, ids = batch[0]
    return features, labels, det, ids

def create_dataloader(path, batch_size=1, shuffle=True, file_list=None):
    """Create a DataLoader over .npz batches.

    batch_size for DataLoader is 1 because each .npz file already contains a batch of samples.
    Optionally pass `file_list` to restrict to a subset of files (useful for train/val splits).
    """
    dataset = AccidentDataset(path, file_list=file_list)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)