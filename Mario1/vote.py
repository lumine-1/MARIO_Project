
import pandas as pd
from collections import Counter

def majority_voting(file1, file2, file3, output_file):
    # 读取 CSV 文件
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)

    # 合并数据框
    merged_df = df1.merge(df2, on='case', suffixes=('_1', '_2'))
    merged_df = merged_df.merge(df3, on='case')

    # 计算多数投票
    def get_majority_vote(row):
        votes = [row['prediction_1'], row['prediction_2'], row['prediction']]
        vote_counts = Counter(votes)
        if vote_counts.most_common(1)[0][1] > 1:  # 检查是否有超过一半的票
            return vote_counts.most_common(1)[0][0]
        else:
            return row['prediction_2']  # 没有多数票时使用

    merged_df['prediction'] = merged_df.apply(get_majority_vote, axis=1)

    # 生成输出数据框
    output_df = merged_df[['case', 'prediction']]

    # 保存为 CSV 文件
    output_df.to_csv(output_file, index=False)


def majority_voting_four(file1, file2, file3, file4, output_file):
    # 读取 CSV 文件
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    df4 = pd.read_csv(file4)

    # 合并数据框
    merged_df = df1.merge(df2, on='case', suffixes=('_1', '_2'))
    merged_df = merged_df.merge(df3, on='case', suffixes=('', '_3'))
    merged_df = merged_df.merge(df4, on='case', suffixes=('', '_4'))

    # 计算多数投票
    def get_majority_vote(row):
        votes = [row['prediction_1'], row['prediction_2'], row['prediction'], row['prediction_4']]
        vote_counts = Counter(votes)
        most_common = vote_counts.most_common()
        if most_common[0][1] > 2:  # 检查是否有超过一半的票（即3票或以上）
            return most_common[0][0]
        else:
            return row['prediction_1']  # 没有多数票时使用第一个文件的预测值

    merged_df['prediction'] = merged_df.apply(get_majority_vote, axis=1)

    # 生成输出数据框
    output_df = merged_df[['case', 'prediction']]

    # 保存为 CSV 文件
    output_df.to_csv(output_file, index=False)

    print(f"结果已保存到 {output_file}")


def compare_predictions(file1, file2):
    # 读取 CSV 文件
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # 合并数据框
    merged_df = df1.merge(df2, on='case', suffixes=('_new', '_original'))

    # 计算 prediction 不同的行数
    diff_count = (merged_df['prediction_new'] != merged_df['prediction_original']).sum()

    return diff_count

# 示例调用
# difference_count = compare_predictions('submission21.csv', 'aug3e10.csv')
# print(f'Number of different predictions: {difference_count}')

# 示例调用
# majority_voting('submission18.csv', 'submission21.csv', 'submission19.2.csv', 'submission18_19.2_21.csv')
# majority_voting_four('heti_3.csv','heti_4.csv', 'submission21.csv',  'all5all.csv','heti_5.csv')








