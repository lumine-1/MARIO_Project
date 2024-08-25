import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
from scipy.stats import rankdata, kendalltau
from torch_lr_finder import TrainDataLoaderIter


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_data(csv_file, random_state=42):
    df = pd.read_csv(csv_file)

    # 计算需要的n_splits
    n_splits = int(1 / 0.25)

    # 创建一个分组
    group_kfold = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # 获取标签和组
    labels = df['label'].values
    groups = df['LOCALIZER'].values

    # 初始化训练和验证索引
    train_indices = []
    val_indices = []

    # 对每个标签分别进行分割
    for label in np.unique(labels):
        label_mask = (labels == label)
        label_groups = groups[label_mask]
        label_indices = np.where(label_mask)[0]

        split_train_indices, split_val_indices = next(
            group_kfold.split(label_indices, labels[label_mask], label_groups))
        train_indices.extend(label_indices[split_train_indices])
        val_indices.extend(label_indices[split_val_indices])

    return df.iloc[train_indices], df.iloc[val_indices]


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss


# 自定义高斯噪声变换
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


class CustomTrainDataLoaderIter(TrainDataLoaderIter):
    def inputs_labels_from_batch(self, batch):
        inputs = batch['image']  # 根据你的字典键名修改
        labels = batch['label']  # 根据你的字典键名修改
        return inputs, labels


def split_data_task1(csv_file, test_size=0.25, random_state=42):
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


# def set_labels_based_on_ratios(a, b, c):
#     """
#     根据每个 LOCALIZER 下图片被预测为 0、1 或 2 的比例设置每个 LOCALIZER 下图片的标签，并保存结果为新的 CSV 文件。
#     """
#
#     data_csv = 'D:/AI_Data/data_2/df_task2_val_challenge.csv'  # 替换为你的数据集文件路径
#     # data_csv = 'test_temp.csv'
#     prediction_csv = 'final2.csv'  # 替换为你的预测文件路径
#     ratio_thresholds = {'0': a, '1': b, '2': c}  # 设置比例阈值
#     output_csv = 'edit_final2.csv'
#
#     # 读取数据集和预测文件
#     data_df = pd.read_csv(data_csv)
#     predictions_df = pd.read_csv(prediction_csv)
#     # 合并数据集和预测数据
#     merged_df = data_df.merge(predictions_df, on='case')
#     # 按 LOCALIZER 分组计算预测比例
#     grouped = merged_df.groupby('LOCALIZER')
#     new_predictions = []
#
#     for localizer, group in grouped:
#         total_images = len(group)
#         prediction_counts = group['prediction'].value_counts(normalize=True).to_dict()
#         count_counts = group['prediction'].value_counts().to_dict()  # 统计各标签个数
#         # 根据比例阈值设置新的标签
#         new_label = None
#         for label, threshold in ratio_thresholds.items():
#             if prediction_counts.get(int(label), 0) > threshold:
#                 new_label = int(label)
#                 break
#
#         # if new_label is None:
#         #     # 如果没有超过任何阈值，仅比较标签0和2的数量
#         #     count_0 = count_counts.get(0, 0)
#         #     count_2 = count_counts.get(2, 0)
#         #     if count_0 > count_2:
#         #         new_label = 0
#         #     elif count_2 > count_0:
#         #         new_label = 2
#         #     else:
#                 # 若数量相等，可以任选一个标签
#                 # new_label = 2
#                 # 保持原来的标签，选择出现频率最高的标签
#                 # new_label = group['prediction'].mode()[0]
#
#         # 更新所有图片的标签
#         group['prediction'] = new_label
#         new_predictions.append(group)
#
#     # 合并更新后的分组数据
#     result_df = pd.concat(new_predictions, ignore_index=True)
#     # 只保留需要的列，并按 case 列排序
#     result_df = result_df[['case', 'prediction']].sort_values(by='case')
#     # 保存新的预测结果文件
#     result_df.to_csv(output_csv, index=False)
#
#     print(f"结果已保存到 {output_csv}")


def set_labels_based_on_ratios(a, b, c):
    """
    根据每个 LOCALIZER 下图片被预测为 0、1 或 2 的比例设置每个 LOCALIZER 下图片的标签，并保存结果为新的 CSV 文件。
    """

    data_csv = 'D:/AI_Data/data_2/df_task2_val_challenge.csv'  # 替换为你的数据集文件路径
    prediction_csv = 'final2.csv'  # 替换为你的预测文件路径
    ratio_thresholds = {'0': a, '1': b, '2': c}  # 设置比例阈值
    output_csv = 'edit_final2.csv'

    # 读取数据集和预测文件
    data_df = pd.read_csv(data_csv)
    predictions_df = pd.read_csv(prediction_csv)

    # 合并数据集和预测数据
    merged_df = data_df.merge(predictions_df, on='case')

    # 保存每个条目的原始标签
    merged_df['original_prediction'] = merged_df['prediction']

    # 按 LOCALIZER 分组计算预测比例
    grouped = merged_df.groupby('LOCALIZER')
    new_predictions = []

    for localizer, group in grouped:
        total_images = len(group)
        prediction_counts = group['prediction'].value_counts(normalize=True).to_dict()
        count_counts = group['prediction'].value_counts().to_dict()  # 统计各标签个数

        # 计算新的标签
        new_label = None
        for label, threshold in ratio_thresholds.items():
            if prediction_counts.get(int(label), 0) > threshold:
                new_label = int(label)
                break

        # 如果 new_label 仍然为 None，则保留原始标签
        if new_label is None:
            group['prediction'] = group['original_prediction']
        else:
            group['prediction'] = new_label

        new_predictions.append(group)

    # 合并更新后的分组数据
    result_df = pd.concat(new_predictions, ignore_index=True)

    # 只保留需要的列，并按 case 列排序
    result_df = result_df[['case', 'prediction']].sort_values(by='case')

    # 保存新的预测结果文件
    result_df.to_csv(output_csv, index=False)

    print(f"结果已保存到 {output_csv}")




def evaluate_predictions(predictions, labels):
    # 读取预测和标签数据
    predictions_df = pd.read_csv(predictions)
    labels_df = pd.read_csv(labels)

    # 合并预测和标签数据
    merged_df = pd.merge(predictions_df, labels_df, left_on='case', right_on='case')

    # 提取真实标签和预测结果
    y_true = merged_df['label'].astype(int)
    y_pred = merged_df['prediction']

    # 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')  # 修改此处，使用'weighted'平均方式
    conf_matrix = confusion_matrix(y_true, y_pred)
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if (
                conf_matrix.shape[0] > 1 and (conf_matrix[0, 0] + conf_matrix[0, 1]) > 0) else 0
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    # 计算排名相关系数
    y_true_ranks = rankdata(y_true)
    y_pred_ranks = rankdata(y_pred)
    rk_correlation, _ = kendalltau(y_true_ranks, y_pred_ranks)
    mean = (f1 + specificity + qwk + rk_correlation)/4

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Specificity: {specificity}")
    print(f"QWK: {qwk}")
    print(f"RK-correlation: {rk_correlation}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Mean:\n{mean}")



def merge():
    df_val = pd.read_csv('D:/AI_Data/data_2/df_task2_val_challenge.csv')
    df_gt = pd.read_csv('D:/AI_Data/data_2/task2_groundtruth.csv')

    # Merge the two dataframes on the 'case' column
    merged_df = pd.merge(df_val, df_gt, on='case')

    # Group by 'case' and aggregate the 'image' and 'LOCALIZER' columns
    final_df = merged_df.groupby('case').agg({
        'id_patient': 'first',
        'side_eye': 'first',
        'BScan': 'first',
        'image': lambda x: ','.join(x),
        'split_type': 'first',
        'LOCALIZER': lambda x: ','.join(x),
        'sex': 'first',
        'age': 'first',
        'num_current_visit': 'first',
        'label': 'first'
    }).reset_index()

    # Save the final dataframe to a CSV file
    final_df.to_csv('final_test.csv', index=False)

    print("Merged file saved as 'final_test.csv'")


def merge_task1():

    df_task1_val = pd.read_csv('D:/AI_Data/data_1/df_task1_val_challenge.csv')
    df_task1_gt = pd.read_csv('D:/AI_Data/data_1/task1_groundtruth.csv')

    # 按照 'case' 列进行合并
    merged_df = pd.merge(df_task1_val, df_task1_gt, on='case', how='inner')

    # 重新排列列顺序
    columns_order = [
        'id_patient', 'side_eye', 'BScan', 'image_at_ti', 'image_at_ti+1',
        'label', 'split_type', 'LOCALIZER_at_ti+1', 'LOCALIZER_at_ti', 'sex',
        'age_at_ti+1', 'age_at_ti', 'num_current_visit_at_i+1',
        'num_current_visit_at_i', 'delta_t', 'case'
    ]
    merged_df = merged_df[columns_order]
    merged_df.to_csv('task1_for2.csv', index=False)


def self_evaluate():
    # 加载预测文件和真实值文件
    pred_df = pd.read_csv('pred.csv')
    gt_df = pd.read_csv('task2_groundtruth.csv')

    # 根据 'case' 进行合并，右连接以确保所有真实值都包括在内
    merged_df = pd.merge(pred_df, gt_df, on='case', how='right')
    merged_df['prediction'] = merged_df['prediction'].fillna(-1).astype(int)  # 如果预测值为空，填充为-1

    # 提取标签和预测
    y_true = merged_df['label'].astype(int)
    y_pred = merged_df['prediction']

    # 计算F1分数
    f1 = f1_score(y_true, y_pred, average='weighted')

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)

    # 计算每个类别的特异性
    specificities = []
    for i in range(len(conf_matrix)):
        # 计算真负类 (TN) 和假正类 (FP)
        tn = np.sum(conf_matrix) - np.sum(conf_matrix[i, :]) - np.sum(conf_matrix[:, i]) + conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)

    # 计算平均特异性
    avg_specificity = np.mean(specificities)

    # 计算加权kappa系数（QWK）
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    # 计算秩相关系数（Kendall's Tau）
    y_true_ranks = rankdata(y_true)
    y_pred_ranks = rankdata(y_pred)
    rk_correlation, _ = kendalltau(y_true_ranks, y_pred_ranks)

    # 计算平均值
    average_metric = (f1 + avg_specificity + qwk + rk_correlation) / 4

    # 输出结果
    print(f"F1 Score: {f1}")
    print(f"Specificity: {avg_specificity}")
    print(f"QWK: {qwk}")
    print(f"Rank Correlation: {rk_correlation}")
    print(f"Average Metric: {average_metric}")
    print(f"Confusion Matrix:\n{conf_matrix}")



if __name__ == "__main__":
    # set_labels_based_on_ratios(0.4, 0.7, 0.3)
    self_evaluate()
    # evaluate_predictions('edit_submission20.1_val.csv', 'test_temp.csv')

    # majority_voting('m1_sub8e21.csv', 'm1_sub8e21g_20_30.csv', 'submission21.csv', 'm1_submission20.csv', 'm1_submission19.csv', 'heti_1.csv')

    # merge_task1()

