from __future__ import absolute_import

import os
from PIL import Image
import numpy as np
import h5py

import torch
from torch.utils.data import Dataset
import random
import pandas as pd
from sklearn.utils import shuffle


def read_image(img_path):
    img = h5py.File(img_path)['data'][()]
    return img


# p should be smaller than 0.5
def dropout_patches_old(feats, p):
    idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0]*(1-p)), replace=False)
    sampled_feats = np.take(feats, idx, axis=0)
    pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0]*p), replace=False)
    pad_feats = np.take(sampled_feats, pad_idx, axis=0)
    sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
    return sampled_feats


def dropout_patches(feats, p):
    idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0]*(1-p)), replace=False)
    sampled_feats = feats[idx, :]
    return sampled_feats


class dataloader(Dataset):
    def __init__(self, dataset, transform=None, training=True, drop_p=0.4, drop_probability=0.6):
        self.dataset = dataset
        self.drop_probability = drop_probability
        self.transform = transform
        self.training = training
        self.drop_p = drop_p

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, label = self.dataset[index]
        feats = torch.load(img_path)
        
        if self.training is True and random.random()>self.drop_probability:
            feats = dropout_patches(feats, self.drop_p)
        length = feats.size(0)

        return feats, label, length


def collate_fn(batch):
    imgs, pids, length = zip(*batch)
    return torch.cat(imgs, dim=0), torch.tensor(pids, dtype=torch.long), length