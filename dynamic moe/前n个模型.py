import pandas as pd

# 读取 Excel 文件
file_path = "D://2//dongmoe2//12//canchafenlei//12hebing.xls"  # 替换为您的文件路径
data = pd.read_excel(file_path)

# 检查列名
print("DataFrame 列名:")
print(data.columns)

# 定义一个函数，用于根据 class 列选择前 n 个模型并计算集成预测值
def calculate_ensemble_prediction(row):
    # 提取 class 值，决定选择的模型数量
    n_models = int(row['class'])  # 根据 class 列决定选择数量

    # 提取概率和预测值
    probabilities = row[['Prob_Class_0', 'Prob_Class_1', 'Prob_Class_2', 'Prob_Class_3']].values
    predictions = row[[0, 1, 2, 3]].values

    # 按概率降序排序
    sorted_indices = probabilities.argsort()[::-1]
    sorted_probabilities = probabilities[sorted_indices]
    sorted_predictions = predictions[sorted_indices]

    # 选择前 n_models 个模型
    selected_predictions = sorted_predictions[:n_models]

    # 如果只选择了一个模型，直接返回该模型的预测值
    if len(selected_predictions) == 1:
        return selected_predictions[0]

    # 如果选择了多个模型，计算平均值
    return sum(selected_predictions) / len(selected_predictions)

# 对每一行应用函数，计算新预测值
data['p'] = data.apply(lambda row: calculate_ensemble_prediction(row), axis=1)

# 创建新的 DataFrame，仅保留 t 列和新预测值 p
output_data = data[['t', 'p']]

# 保存结果到新的 Excel 文件
output_file_path = "D://2//dongmoe2//12//canchafenlei//12result.xls"  # 替换为您想要保存的路径
output_data.to_excel(output_file_path, index=False)

print(f"结果已保存到 {output_file_path}")