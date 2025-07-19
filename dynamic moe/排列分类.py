import pandas as pd

# 读取CSV文件
file_path = 'D://2//dongmoe2//6//cancha//cancha.xls'  # 替换为您的文件路径
data = pd.read_excel(file_path)

# 确保数据列名为 'wl'
if 'ave' not in data.columns:
    raise ValueError("CSV 文件中缺少 'ave' 列，请检查文件格式。")

# 保存原始索引
data['original_index'] = data.index

# 对全量数据进行排序以计算分界线
sorted_data = data.sort_values(by='ave').reset_index(drop=True)  # 升序排序

# 确定高、中、低流量的分界线
low_threshold = sorted_data['ave'].quantile(0.5)   #0.3
high_threshold = sorted_data['ave'].quantile(0.7)  #0.7

# 定义分类函数
def classify(value):
    if value <= low_threshold:
        return 1  # 低残差
    elif value <= high_threshold:
        return 2  # 中残差
    else:
        return 3  # 高残差

# 给数据添加类别（使用分界线）
data['class'] = data['ave'].apply(classify)

# 恢复原始顺序
data = data.sort_values(by='original_index').drop(columns=['original_index'])

# 输出到新CSV文件，并确保没有空行
output_path = 'D://2//dongmoe2//6//cancha//canchaf.xls'
data.to_excel(output_path, index=False)

print(f"处理完成，新文件已保存为 '{output_path}'")