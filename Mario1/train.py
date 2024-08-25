import copy

import torch
from sklearn.metrics import accuracy_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import transforms,AutoAugment, AutoAugmentPolicy
from tqdm import tqdm

from OCTData import TrainData, TrainData_aug
from utils import calculate_metrics, DualImageTransform, AddGaussianNoise


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scaler = GradScaler()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Training phase
        model.train()

        running_loss = 0.0
        all_labels = []
        all_preds = []

        train_loader_tqdm = tqdm(train_loader, desc="Training")

        for samples in train_loader_tqdm:
            inputs_ti = samples['image_ti'].to(device, dtype=torch.float)
            inputs_ti_1 = samples['image_ti_1'].to(device, dtype=torch.float)
            labels = samples['label'].to(device, dtype=torch.long)  # 确保标签是1D张量

            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs_ti, inputs_ti_1)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs_ti.size(0)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)

        print(f'Loss: {epoch_loss:.4f}')

        train_accuracy, train_f1, train_spearman, train_specificity, train_mean_metrics, train_cm = calculate_metrics(all_labels, all_preds)
        print(f'Train Accuracy: {train_accuracy:.4f} F1 Score: {train_f1:.4f} Spearman Corr: {train_spearman:.4f} Specificity: {train_specificity:.4f} Mean Metrics: {train_mean_metrics:.4f}')
        print(f'Train Confusion Matrix:\n{train_cm}')

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_labels = []
        val_preds = []

        val_loader_tqdm = tqdm(val_loader, desc="Validation")

        for samples in val_loader_tqdm:
            inputs_ti = samples['image_ti'].to(device, dtype=torch.float)
            inputs_ti_1 = samples['image_ti_1'].to(device, dtype=torch.float)
            labels = samples['label'].to(device, dtype=torch.long)  # 确保标签是1D张量

            with torch.no_grad():
                with autocast():
                    outputs = model(inputs_ti, inputs_ti_1)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs_ti.size(0)
                _, preds = torch.max(outputs, 1)

                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)

        print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')

        val_accuracy, val_f1, val_spearman, val_specificity, val_mean_metrics, val_cm = calculate_metrics(val_labels, val_preds)
        print(f'Validation Accuracy: {val_accuracy:.4f} F1 Score: {val_f1:.4f} Spearman Corr: {val_spearman:.4f} Specificity: {val_specificity:.4f} Mean Metrics: {val_mean_metrics:.4f}')
        print(f'Validation Confusion Matrix:\n{val_cm}')

        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), f'model_s17_{epoch + 1}.pth')
            print(f'Model saved at epoch {epoch + 1}')

    return model



augmentation = transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.2),
        transforms.RandomAffine(degrees=20, scale=(0.9, 1.1), translate=(0.05, 0.05)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomApply([AddGaussianNoise(0., 0.05)], p=0.1)
    ])


# augmentation = transforms.Compose([
#         AutoAugment(policy=AutoAugmentPolicy.IMAGENET),  # 使用 ImageNet 策略
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5], std=[0.5]),
#     ])



def train_model_aug(model, train_csv, val_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scaler = GradScaler()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Training phase
        model.train()

        running_loss = 0.0
        all_labels = []
        all_preds = []

        dir = 'D:/AI_Data/data_1/train'
        dual_augmentation = DualImageTransform(augmentation)
        train_dataset = TrainData_aug(csv_file=train_csv, root_dir=dir, transform=dual_augmentation)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        train_loader_tqdm = tqdm(train_loader, desc="Training")

        for samples in train_loader_tqdm:
            inputs_ti = samples['image_ti'].to(device, dtype=torch.float)
            inputs_ti_1 = samples['image_ti_1'].to(device, dtype=torch.float)
            labels = samples['label'].to(device, dtype=torch.long)  # 确保标签是1D张量

            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs_ti, inputs_ti_1)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs_ti.size(0)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)

        print(f'Loss: {epoch_loss:.4f}')

        train_accuracy, train_f1, train_spearman, train_specificity, train_mean_metrics, train_cm = calculate_metrics(all_labels, all_preds)
        print(f'Train Accuracy: {train_accuracy:.4f} F1 Score: {train_f1:.4f} Spearman Corr: {train_spearman:.4f} Specificity: {train_specificity:.4f} Mean Metrics: {train_mean_metrics:.4f}')
        print(f'Train Confusion Matrix:\n{train_cm}')

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_labels = []
        val_preds = []

        val_loader_tqdm = tqdm(val_loader, desc="Validation")

        for samples in val_loader_tqdm:
            inputs_ti = samples['image_ti'].to(device, dtype=torch.float)
            inputs_ti_1 = samples['image_ti_1'].to(device, dtype=torch.float)
            labels = samples['label'].to(device, dtype=torch.long)  # 确保标签是1D张量

            with torch.no_grad():
                with autocast():
                    outputs = model(inputs_ti, inputs_ti_1)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs_ti.size(0)
                _, preds = torch.max(outputs, 1)

                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)

        print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')

        val_accuracy, val_f1, val_spearman, val_specificity, val_mean_metrics, val_cm = calculate_metrics(val_labels, val_preds)
        print(f'Validation Accuracy: {val_accuracy:.4f} F1 Score: {val_f1:.4f} Spearman Corr: {val_spearman:.4f} Specificity: {val_specificity:.4f} Mean Metrics: {val_mean_metrics:.4f}')
        print(f'Validation Confusion Matrix:\n{val_cm}')

        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), f'model_s24_{epoch + 1}.pth')
            print(f'Model saved at epoch {epoch + 1}')

    return model









