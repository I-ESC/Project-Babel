# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, split=None, train_data=None, val_data=None, val_labels=None, test_data=None, test_labels=None):
        self.split = split
        self.train_data = train_data
        self.val_data = val_data
        self.val_labels = val_labels
        self.test_data = test_data
        self.test_labels = test_labels

    def __getitem__(self, i):
        if self.split == 'train':
            num = np.random.randint(len(self.train_data[i]))
            rgb_data, imu_data, skel_data = self.train_data[i][num]
            return rgb_data, imu_data, skel_data
        elif self.split == 'val':
            rgb_data, imu_data, skel_data = self.val_data[i]
            labels = self.val_labels[i]
            return rgb_data, imu_data, skel_data, labels
        elif self.split == 'test':
            rgb_data, imu_data, skel_data = self.test_data[i]
            labels = self.test_labels[i]
            return rgb_data, imu_data, skel_data, labels
        else:
            raise ValueError("Invalid split. Expected one of: 'train', 'val', 'test'")

    def __len__(self):
        if self.split == 'train':
            return len(self.train_data)
        elif self.split == 'val':
            return len(self.val_data)
        elif self.split == 'test':
            return len(self.test_data)
        else:
            raise ValueError("Invalid split. Expected one of: 'train', 'val', 'test'")