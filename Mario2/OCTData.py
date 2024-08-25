import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class OCTData(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['image'])

        image = Image.open(img_name).convert('L')

        if self.transform:
            image = self.transform(image)

        sample = {
            'image': image,
            'label': int(self.data_frame.iloc[idx]['label']),
            'case_id': self.data_frame.iloc[idx]['case'],
            'LOCALIZER': self.data_frame.iloc[idx]['LOCALIZER'],
        }

        return sample


class OCTData_task1(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 获取标签
        label = int(self.data_frame.iloc[idx, 5])  # 确保标签是一个整数
        # 如果标签为3，跳过此样本，获取下一个样本
        if label == 3:
            if idx + 1 < len(self.data_frame):
                return self.__getitem__(idx + 1)
            else:
                raise IndexError("All remaining items have label 3")
        img_name_ti = os.path.join(self.root_dir, self.data_frame.iloc[idx, 3])
        image_ti = Image.open(img_name_ti).convert('L')  # 灰度图
        if self.transform:
            image_ti = self.transform(image_ti)
        sample = {
            'image': image_ti,
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
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['image'])

        image = Image.open(img_name).convert('L')

        if self.transform:
            image = self.transform(image)

        sample = {
            'image': image,
            'case_ids': self.data_frame.iloc[idx]['case']
        }

        return sample



class OCTDataC(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.grouped_data = self.data_frame.groupby('LOCALIZER')

    def __len__(self):
        return len(self.grouped_data)

    def __getitem__(self, idx):
        localizer = list(self.grouped_data.groups.keys())[idx]
        group = self.grouped_data.get_group(localizer)

        images = []
        for _, row in group.iterrows():
            img_name = os.path.join(self.root_dir, row['image'])
            image = Image.open(img_name).convert('L')
            if self.transform:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images)  # 将多个图像堆叠成一个批次

        label = torch.tensor(group['label'].values[0], dtype=torch.long)  # 所有图像的标签相同
        sample = {
            'image': images,
            'label': label,
            'case_id': group['case'].values[0],
            'LOCALIZER': localizer,
        }

        return sample


# Ctest
class TestDataC(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.grouped_data = self.data_frame.groupby('LOCALIZER')

    def __len__(self):
        return len(self.grouped_data)

    def __getitem__(self, idx):
        localizer = list(self.grouped_data.groups.keys())[idx]
        group = self.grouped_data.get_group(localizer)

        images = []
        case_ids = []
        for _, row in group.iterrows():
            img_name = os.path.join(self.root_dir, row['image'])
            image = Image.open(img_name).convert('L')
            if self.transform:
                image = self.transform(image)
            images.append(image)
            case_ids.append(row['case'])

        images = torch.stack(images)  # 将多个图像堆叠成一个批次

        sample = {
            'image': images,
            'case_ids': case_ids,
            'LOCALIZER': localizer,
        }

        return sample

