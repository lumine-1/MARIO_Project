import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, cohen_kappa_score
from torchvision.transforms import transforms

from OCTData import OCTData, OCTDataC, OCTData_task1
from utils import AddGaussianNoise
from scipy.stats import rankdata, kendalltau
from torch.cuda.amp import autocast, GradScaler


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()

        running_loss = 0.0
        running_corrects = 0
        all_labels = []
        all_preds = []

        # Adding tqdm progress bar for training
        train_loader_tqdm = tqdm(train_loader, desc="Training")

        for samples in train_loader_tqdm:
            inputs = samples['image'].to(device, dtype=torch.float)
            labels = samples['label'].to(device, dtype=torch.long)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}')

        # 打印训练集的混淆矩阵
        train_conf_matrix = confusion_matrix(all_labels, all_preds)
        print(f'Training Confusion Matrix:\n{train_conf_matrix}')

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_labels = []
        val_preds = []

        # Adding tqdm progress bar for validation
        val_loader_tqdm = tqdm(val_loader, desc="Validation")

        for samples in val_loader_tqdm:
            inputs = samples['image'].to(device, dtype=torch.float)
            labels = samples['label'].to(device, dtype=torch.long)

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} Val F1: {val_f1:.4f}')

        # 打印验证集的混淆矩阵
        val_conf_matrix = confusion_matrix(val_labels, val_preds)
        print(f'Validation Confusion Matrix:\n{val_conf_matrix}')

        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), f'model_s19_{epoch + 1}.pth')
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

transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def train_model_aug(model, csv, val_loader, criterion, optimizer, num_epochs=3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dir = 'D:/AI_Data/data2_aug2/augmented_train'

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

    # Scaler for mixed precision training
    scaler = GradScaler()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()

        running_loss = 0.0
        running_corrects = 0
        all_labels = []
        all_preds = []

        train_dataset = OCTData(csv_file=csv, root_dir=dir, transform=augmentation)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        train_loader_tqdm = tqdm(train_loader, desc="Training")

        for samples in train_loader_tqdm:
            inputs = samples['image'].to(device, dtype=torch.float)
            labels = samples['label'].to(device, dtype=torch.long)

            optimizer.zero_grad()

            with autocast():  # Enables mixed precision training
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        train_conf_matrix = confusion_matrix(all_labels, all_preds)
        specificity = train_conf_matrix[0, 0] / (train_conf_matrix[0, 0] + train_conf_matrix[0, 1]) if (
                    train_conf_matrix.shape[0] > 1 and (train_conf_matrix[0, 0] + train_conf_matrix[0, 1]) > 0) else 0
        qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        y_true_ranks = rankdata(all_labels)
        y_pred_ranks = rankdata(all_preds)
        rk_correlation, _ = kendalltau(y_true_ranks, y_pred_ranks)
        mean = (specificity + qwk + rk_correlation + epoch_f1) / 4

        print(
            f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f} Specificity: {specificity:.4f} QWK: {qwk:.4f} RK-correlation: {rk_correlation:.4f}')
        print(f'Training Confusion Matrix:\n{train_conf_matrix}')
        print(f'Avg Metric: {mean:.4f}')

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_labels = []
        val_preds = []

        val_loader_tqdm = tqdm(val_loader, desc="Validation")

        for samples in val_loader_tqdm:
            inputs = samples['image'].to(device, dtype=torch.float)
            labels = samples['label'].to(device, dtype=torch.long)

            with torch.no_grad(), autocast():  # Enables mixed precision evaluation
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        val_conf_matrix = confusion_matrix(val_labels, val_preds)
        val_specificity = val_conf_matrix[0, 0] / (val_conf_matrix[0, 0] + val_conf_matrix[0, 1]) if (
                    val_conf_matrix.shape[0] > 1 and (val_conf_matrix[0, 0] + val_conf_matrix[0, 1]) > 0) else 0
        val_qwk = cohen_kappa_score(val_labels, val_preds, weights='quadratic')
        val_y_true_ranks = rankdata(val_labels)
        val_y_pred_ranks = rankdata(val_preds)
        val_rk_correlation, _ = kendalltau(val_y_true_ranks, val_y_pred_ranks)
        val_mean = (val_specificity + val_qwk + val_rk_correlation + val_f1) / 4

        print(
            f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} Val F1: {val_f1:.4f} Val Specificity: {val_specificity:.4f} Val QWK: {val_qwk:.4f} Val RK-correlation: {val_rk_correlation:.4f}')
        print(f'Validation Confusion Matrix:\n{val_conf_matrix}')
        print(f'Avg Metric: {val_mean:.4f}')

        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), f'final_2_{epoch + 1}.pth')
            print(f'Model saved at epoch {epoch + 1}')

    return model


# 无验证
# def train_model_aug(model, csv, val_loader, criterion, optimizer, num_epochs=3):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#
#     dir = 'D:/AI_Data/data2_aug2/augmented_train'
#
#     # Learning rate scheduler
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
#
#     # Initialize GradScaler for mixed precision training
#     scaler = GradScaler()
#
#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch}/{num_epochs - 1}')
#         print('-' * 10)
#
#         model.train()
#
#         running_loss = 0.0
#         running_corrects = 0
#         all_labels = []
#         all_preds = []
#
#         train_dataset = OCTData(csv_file=csv, root_dir=dir, transform=augmentation)
#         train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
#         # Adding tqdm progress bar for training
#         train_loader_tqdm = tqdm(train_loader, desc="Training")
#
#         for samples in train_loader_tqdm:
#             inputs = samples['image'].to(device, dtype=torch.float)
#             labels = samples['label'].to(device, dtype=torch.long)
#
#             optimizer.zero_grad()
#
#             # Mixed precision training
#             with autocast():
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#
#             _, preds = torch.max(outputs, 1)
#
#             # Scale the loss and call backward() to create scaled gradients
#             scaler.scale(loss).backward()
#
#             # Unscale gradients and call optimizer step() to update weights
#             scaler.step(optimizer)
#
#             # Update the scale for next iteration
#             scaler.update()
#
#             running_loss += loss.item() * inputs.size(0)
#             running_corrects += torch.sum(preds == labels.data)
#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(preds.cpu().numpy())
#
#         scheduler.step()
#         epoch_loss = running_loss / len(train_loader.dataset)
#         epoch_acc = running_corrects.double() / len(train_loader.dataset)
#         epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
#
#         # Calculate additional metrics
#         train_conf_matrix = confusion_matrix(all_labels, all_preds)
#         specificity = train_conf_matrix[0, 0] / (train_conf_matrix[0, 0] + train_conf_matrix[0, 1]) if (
#                     train_conf_matrix.shape[0] > 1 and (train_conf_matrix[0, 0] + train_conf_matrix[0, 1]) > 0) else 0
#         qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
#         y_true_ranks = rankdata(all_labels)
#         y_pred_ranks = rankdata(all_preds)
#         rk_correlation, _ = kendalltau(y_true_ranks, y_pred_ranks)
#         mean = (specificity + qwk + rk_correlation + epoch_f1) / 4
#
#         print(
#             f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f} Specificity: {specificity:.4f} QWK: {qwk:.4f} RK-correlation: {rk_correlation:.4f}')
#         print(f'Training Confusion Matrix:\n{train_conf_matrix}')
#         print(f'Avg Metric: {mean:.4f}')
#
#         if (epoch + 1) % 1 == 0:
#             torch.save(model.state_dict(), f'model_s24_{epoch + 1}.pth')
#             print(f'Model saved at epoch {epoch + 1}')
#
#     return model



