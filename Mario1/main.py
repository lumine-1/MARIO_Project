import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, random_split

from OCTData import TrainData
from evaluate import evaluate_model
from model import ChangeFormer, SiameseNetwork
from train import train_model, train_model_aug
from utils import set_seed, split_data
from torchvision import transforms


def main():
    set_seed(42)  # 你可以选择任何整数作为种子

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    csv = 'D:/AI_Data/data_1/df_task1_train_challenge.csv'
    dir = 'D:/AI_Data/data_1/train'
    testdir = 'D:/AI_Data/data_1/val'
    final_test_csv = 'final_test.csv'

    # 按比例分割数据集
    # labels = [dataset[i]['label'] for i in range(len(dataset))]
    # train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=0.25, stratify=labels, random_state=42)
    # train_dataset = Subset(dataset, train_idx)
    # val_dataset = Subset(dataset, val_idx)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 按比例和限定分割数据集
    train_df, test_df = split_data(csv)
    train_csv_file = 'train_temp.csv'
    test_csv_file = 'test_temp.csv'
    train_df.to_csv(train_csv_file, index=False)
    test_df.to_csv(test_csv_file, index=False)

    # train_dataset = TrainData(csv_file=train_csv_file, root_dir=dir, transform=transform)     #正常训练
    # train_dataset = TrainData(csv_file=csv, root_dir=dir, transform=transform)    # 训练全部
    # train_loader = DataLoader(train_dataset, batch_size=16,shuffle=True)

    val_dataset = TrainData(csv_file=final_test_csv, root_dir=testdir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 模型
    model = SiameseNetwork()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 损失函数，策略
    # criterion = nn.CrossEntropyLoss()
    weights = torch.tensor([1.5, 1.0, 1.5, 2.0], dtype=torch.float)
    weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    # 训练模型
    train_model_aug(model, csv, val_loader, criterion, optimizer, num_epochs=30)     # 加强训练
    # train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30)
    # torch.save(model.state_dict(), 'sub21.2.pth')

    # 加载权重
    # checkpoint_path = 'D:/AI_Data/PyCharm/Mario1/record/submission21/model_s21_2_18.pth'  # 替换为实际的模型权重文件路径
    # model.load_state_dict(torch.load(checkpoint_path))

    # 评估
    evaluate_model(model, 16, "submission24.1.csv")




if __name__ == '__main__':
    main()




