import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class TrainData(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name_ti = os.path.join(self.root_dir, self.data_frame.iloc[idx, 3])
        img_name_ti_1 = os.path.join(self.root_dir, self.data_frame.iloc[idx, 4])

        image_ti = Image.open(img_name_ti).convert('L')  # 灰度图
        image_ti_1 = Image.open(img_name_ti_1).convert('L')  # 灰度图

        if self.transform:
            image_ti = self.transform(image_ti)
            image_ti_1 = self.transform(image_ti_1)

        label = int(self.data_frame.iloc[idx, 5])  # 确保标签是一个整数

        sample = {
            'image_ti': image_ti,
            'image_ti_1': image_ti_1,
            'label': label
        }

        return sample


class TestData(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name_ti = os.path.join(self.root_dir, self.data_frame.iloc[idx, 3])
        img_name_ti_1 = os.path.join(self.root_dir, self.data_frame.iloc[idx, 4])

        image_ti = Image.open(img_name_ti).convert('L')  # 灰度图
        image_ti_1 = Image.open(img_name_ti_1).convert('L')  # 灰度图

        if self.transform:
            image_ti = self.transform(image_ti)
            image_ti_1 = self.transform(image_ti_1)

        sample = {
            'image_ti': image_ti,
            'image_ti_1': image_ti_1,
            'case_id': self.data_frame.iloc[idx, 14],
        }

        return sample


class TrainData_aug(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name_ti = os.path.join(self.root_dir, self.data_frame.iloc[idx, 3])
        img_name_ti_1 = os.path.join(self.root_dir, self.data_frame.iloc[idx, 4])
        image_ti = Image.open(img_name_ti).convert('L')  # 灰度图
        image_ti_1 = Image.open(img_name_ti_1).convert('L')  # 灰度图
        if self.transform:
            image_ti, image_ti_1 = self.transform(image_ti, image_ti_1)
        label = int(self.data_frame.iloc[idx, 5])  # 确保标签是一个整数

        sample = {
            'image_ti': image_ti,
            'image_ti_1': image_ti_1,
            'label': label
        }

        return sample













