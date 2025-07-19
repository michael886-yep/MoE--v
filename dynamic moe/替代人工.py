import pandas as pd

# 文件路径
file_1_path = "D://2//dongmoe2//12//canchafenlei//12probabilities.xls"  # 替换为 1.xls 的实际路径
file_2_path = "D://2//dongmoe2//12//canchafenlei//12canval.xls"  # 替换为 2.xls 的实际路径
file_3_path = "D://2//dongmoe2//12//canchafenlei//12hebing.xls"  # 替换为 3.xls 的实际路径
output_path = "D://2//dongmoe2//12//canchafenlei//12hebing.xls"  # 替换为输出文件的路径

# 读取 Excel 文件
data_1 = pd.read_excel(file_1_path)
data_2 = pd.read_excel(file_2_path)
data_3 = pd.read_excel(file_3_path)

# 确保列名存在
required_columns_1 = ['Prob_Class_0', 'Prob_Class_1', 'Prob_Class_2', 'Prob_Class_3']
required_columns_2 = ['class']

for col in required_columns_1:
    if col not in data_1.columns:
        raise ValueError(f"1.xls 中缺少列: {col}")

if 'class' not in data_2.columns:
    raise ValueError("2.xls 中缺少列: class")

for col in required_columns_1:
    if col not in data_3.columns:
        raise ValueError(f"3.xls 中缺少列: {col}")

if 'class' not in data_3.columns:
    raise ValueError("3.xls 中缺少列: class")

# 替换 3.xls 中的列
data_3[required_columns_1] = data_1[required_columns_1]
data_3['class'] = data_2['class']

# 保存到新的文件
data_3.to_excel(output_path, index=False)

print(f"替换完成，结果已保存到 {output_path}")