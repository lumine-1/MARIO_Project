import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models

from OCTData import OCTData, TestData, OCTDataC, TestDataC, OCTData_task1
from evaluate import evaluate_model
from models import OCTResNet101, OCTResNet101_n, ConvNeXtLarge_attention, OCTResNet50C, OCTConvNeXtC
# from models import OCTEfficientNet
from train import train_model, train_model_aug
from utils import set_seed, split_data, FocalLoss, CustomTrainDataLoaderIter, split_data_task1
from torch_lr_finder import LRFinder


def main():
    set_seed(42)  # 你可以选择任何整数作为种子

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # dir_train = 'D:/AI_Data/data2_aug2/augmented_train'
    # dir_test = 'D:/AI_Data/data2_aug/augmented_train'

    # 原来的训练方式
    csv = 'D:/AI_Data/data2_aug2/augmented_train.csv'
    dir = 'D:/AI_Data/data2_aug2/augmented_train'
    testdir = 'D:/AI_Data/data_2/val'
    final_test_csv = 'D:/AI_Data/data_2/final_test.csv'

    #使用数据集1训练
    # csv = 'D:/AI_Data/data_1/task1_for2.csv'
    # dir = 'D:/AI_Data/data_1/all'

    # 按比例分割数据集
    train_df, test_df = split_data(csv)
    train_csv_file = 'train_temp.csv'
    test_csv_file = 'test_temp.csv'
    train_df.to_csv(train_csv_file, index=False)
    test_df.to_csv(test_csv_file, index=False)

    # 正常训练
    train_dataset = OCTData(csv_file=csv, root_dir=dir, transform=transform)
    val_dataset = OCTData(csv_file=final_test_csv, root_dir=testdir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # 变为整体
    # train_dataset = OCTDataC(csv_file=train_csv_file, root_dir=dir, transform=transform)
    # val_dataset = OCTDataC(csv_file=final_test_csv, root_dir=testdir, transform=transform)
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # model
    # model = ConvNeXtLarge_attention()
    # checkpoint_path = 'model_s23_49.pth'  # 替换为实际的模型权重文件路径
    # model.load_state_dict(torch.load(checkpoint_path))

    # 整体模型
    model = ConvNeXtLarge_attention()
    ###########
    checkpoint_path = 'final_2_6.pth'  # 替换为实际的模型权重文件路径
    model.load_state_dict(torch.load(checkpoint_path))
    ###########
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 1 创建加权的 CrossEntropyLoss
    weights = torch.tensor([1.5, 1.0, 2.0], dtype=torch.float)
    weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # 训练模型
    optimizer = optim.AdamW(model.parameters(), lr=0.000005)
    # train_model_aug(model, csv, val_loader, criterion, optimizer, 50)

    # 保存模型
    # torch.save(model.state_dict(), 'model_23.pth')

    checkpoint_path = 'final_2_6.pth'  # 替换为实际的模型权重文件路径
    model.load_state_dict(torch.load(checkpoint_path))

    test_dataset = TestData(csv_file='D:/AI_Data/data_2/df_task2_val_challenge.csv',
                                    root_dir='D:/AI_Data/data_2/val', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    evaluate_model(model, test_loader, 'final2.csv')



    # # 删除临时文件
    # # os.remove(train_csv_file)
    # # os.remove(test_csv_file)

    # 使用 Learning Rate Finder
    # lr_finder = LRFinder(model, optimizer, criterion, device="cuda" if torch.cuda.is_available() else "cpu")
    # train_loader_iter = CustomTrainDataLoaderIter(train_loader)
    # lr_finder.range_test(train_loader_iter, end_lr=0.001, num_iter=1000)
    # lr_finder.plot()  # 检查损失-学习率图
    # best_lr = lr_finder.suggest_lr()
    # print(f"Suggested Learning Rate: {best_lr}")



if __name__ == "__main__":
    main()



