import random

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metrics(labels, preds):
    labels = np.asarray(labels).flatten()
    preds = np.asarray(preds).flatten()

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    spearman_corr, _ = spearmanr(labels, preds)
    cm = confusion_matrix(labels, preds)

    # 计算每个类别的特异性
    specificities = []
    for i in range(cm.shape[0]):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)

    # 计算平均特异性
    average_specificity = np.mean(specificities)

    mean_metrics = (accuracy + f1 + spearman_corr + average_specificity) / 4

    return accuracy, f1, spearman_corr, average_specificity, mean_metrics, cm


def split_data(csv_file, test_size=0.25, random_state=42):
    df = pd.read_csv(csv_file)

    # 创建联合组
    groups = df[['LOCALIZER_at_ti', 'LOCALIZER_at_ti+1']].astype(str).agg('_'.join, axis=1).values

    # 获取标签
    labels = df['label'].values

    # 创建 StratifiedGroupKFold 分割器
    n_splits = int(1 / test_size)
    group_kfold = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # 初始化训练和验证索引
    train_indices, val_indices = next(group_kfold.split(df, labels, groups))

    return df.iloc[train_indices], df.iloc[val_indices]


# 不同变换
# class DualImageTransform:
#     def __init__(self, transform):
#         self.transform = transform
#
#     def __call__(self, image_ti, image_ti_1):
#         image_ti = self.transform(image_ti)
#         image_ti_1 = self.transform(image_ti_1)
#         return image_ti, image_ti_1

# 相同变换
class DualImageTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image_ti, image_ti_1):
        seed = random.randint(0, 2 ** 32)  # 生成随机种子
        random.seed(seed)
        torch.manual_seed(seed)
        image_ti = self.transform(image_ti)
        random.seed(seed)
        torch.manual_seed(seed)
        image_ti_1 = self.transform(image_ti_1)

        return image_ti, image_ti_1


# 自定义高斯噪声变换
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


def merge_test():
    # 加载两个 CSV 文件
    df_task1 = pd.read_csv('df_task1_val_challenge.csv')
    df_groundtruth = pd.read_csv('task1_groundtruth.csv')

    # 合并两个数据框，基于 'case' 列
    merged_df = pd.merge(df_task1, df_groundtruth, on='case')

    # 重新排列列顺序
    columns_order = [
        'id_patient', 'side_eye', 'BScan', 'image_at_ti', 'image_at_ti+1',
        'label', 'split_type', 'LOCALIZER_at_ti+1', 'LOCALIZER_at_ti',
        'sex', 'age_at_ti+1', 'age_at_ti', 'num_current_visit_at_i+1',
        'num_current_visit_at_i', 'delta_t', 'case'
    ]
    merged_df = merged_df[columns_order]

    # 保存合并后的文件
    merged_df.to_csv('merged_task1.csv', index=False)

    print("合并后的文件已保存为 'merged_task1.csv'")


def self_evaluate(pred_file, groundtruth_file):
    # 加载预测文件和真实标签文件
    pred_df = pd.read_csv(pred_file)
    groundtruth_df = pd.read_csv(groundtruth_file)

    # 合并数据，以便基于case匹配真实标签和预测结果
    merged_df = pd.merge(groundtruth_df, pred_df, on='case')

    # 提取标签和预测
    labels = merged_df['label'].values
    preds = merged_df['prediction'].values

    # 使用calculate_metrics评估预测
    accuracy, f1, spearman_corr, avg_specificity, mean_metrics, cm = calculate_metrics(labels, preds)

    # 打印结果
    print(f"Accuracy: {accuracy:.5f}")
    print(f"F1 Score: {f1:.5f}")
    print(f"Spearman Correlation: {spearman_corr:.5f}")
    print(f"Average Specificity: {avg_specificity:.5f}")
    print(f"Mean of Metrics: {mean_metrics:.5f}")
    print(f"Confusion Matrix:\n{cm}")

    return accuracy, f1, spearman_corr, avg_specificity, mean_metrics, cm





if __name__ == "__main__":
    # merge_test()
    self_evaluate('pred.csv', 'task1_groundtruth.csv')








