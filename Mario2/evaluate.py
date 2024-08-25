import pandas as pd
import torch


def evaluate_model(model, test_loader, output_file='submission.csv'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions = []

    with torch.no_grad():
        for samples in test_loader:
            inputs = samples['image'].to(device)
            case_ids = samples['case_ids']

            outputs = model(inputs)

            # 假设模型输出是一个包含4个类别概率的张量
            _, predicted = torch.max(outputs, 1)

            # 将预测结果和case_id保存
            for case_id, prediction in zip(case_ids, predicted):
                predictions.append({'case': case_id.item(), 'prediction': prediction.item()})

            # 生成提交文件
        df = pd.DataFrame(predictions)
        df.to_csv(output_file, index=False)

        print(f'Submission file saved as {output_file}')


