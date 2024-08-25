from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from OCTData import TestData


def evaluate(model, test_loader, output_file='submission.csv'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions = []

    with torch.no_grad():
        for samples in test_loader:
            inputs_ti = samples['image_ti'].to(device)
            inputs_ti_1 = samples['image_ti_1'].to(device)
            case_ids = samples['case_id']

            outputs = model(inputs_ti, inputs_ti_1)

            # 假设模型输出是一个包含4个类别概率的张量
            _, predicted = torch.max(outputs, 1)

            # 将预测结果和case_id保存
            for case_id, prediction in zip(case_ids, predicted):
                predictions.append({'case': case_id.item(), 'prediction': prediction.item()})

            # 生成提交文件
        df = pd.DataFrame(predictions)
        df.to_csv(output_file, index=False)

        print(f'Submission file saved as {output_file}')


transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def evaluate_model(model, batch, output_file='submission.csv'):
    test_dataset = TestData(csv_file='D:/AI_Data/data_1/df_task1_val_challenge.csv',
                                    root_dir='D:/AI_Data/data_1/val', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    evaluate(model, test_loader, output_file=output_file)
