import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

# 读取提交的预测概率 (submission.csv)
submission_df = pd.read_csv('submission_final.csv')

# 读取真实标签 (solution.csv)
solution_df = pd.read_csv('test.csv')

# 合并两个数据框，确保id列匹配
merged_df = pd.merge(submission_df, solution_df, on='id', suffixes=('_pred', '_true'))

# 提取预测概率列和真实标签列
y_pred = merged_df[['winner_model_a_pred', 'winner_model_b_pred', 'winner_tie_pred']].values
y_true = merged_df[['winner_model_a_true', 'winner_model_b_true', 'winner_tie_true']].values

# 将 one-hot 真实标签转换为类别索引 (1D 数组, shape: [num_samples])
y_true_labels = np.argmax(y_true, axis=1)

# 获取 y_pred 最大概率的类别索引 (1D 数组, shape: [num_samples])
y_pred_labels = np.argmax(y_pred, axis=1)

# 计算 Accuracy
accuracy = np.mean(y_pred_labels == y_true_labels)

print(f"Accuracy: {accuracy:.4f}")

# 计算交叉熵损失 (log loss)
loss = log_loss(y_true, y_pred)

print(f"Log Loss: {loss}")


