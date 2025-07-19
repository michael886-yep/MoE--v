import pandas as pd

# 文件路径
file1_path = "D://2//fenceng//3//jieguo//3lowval2.xls"  # 替换为您的第一个 xls 文件路径
file2_path = "D://2//fenceng//3//jieguo//3middleval2.xls"  # 替换为您的第二个 xls 文件路径
file3_path = "D://2//fenceng//3//jieguo//3highval2.xls"  # 替换为您的第三个 xls 文件路径

# 输出文件路径
output_file_path = "D://2//fenceng//3//jieguo//3merge.xls"  # 替换为输出文件路径

# 读取三个 xls 文件
data1 = pd.read_excel(file1_path)
data2 = pd.read_excel(file2_path)
data3 = pd.read_excel(file3_path)

# 合并三个数据集
merged_data = pd.concat([data1, data2, data3], ignore_index=True)

# 根据 id 列进行升序排序
merged_data_sorted = merged_data.sort_values(by='id', ascending=True)

# 输出为新的 Excel 文件
merged_data_sorted.to_excel(output_file_path, index=False)

print(f"处理完成，合并并排序后的文件已保存为 {output_file_path}")