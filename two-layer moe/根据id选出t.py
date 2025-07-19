import pandas as pd

# 文件路径
low_file_path = "D://2//fenceng//3//jieguo//3lowval.xls"  # 替换为3lowval.xls的实际路径
middle_file_path = "D://2//fenceng//3//jieguo//3middleval.xls"  # 替换为3middleval.xls的实际路径
high_file_path = "D://2//fenceng//3//jieguo//3highval.xls"  # 替换为3highval.xls的实际路径
ml_file_path = "D://2//fenceng//3//jieguo//ml.xls"  # 替换为ml.xls的实际路径

# 输出文件路径
output_low_path = "D://2//fenceng//3//jieguo//3lowval1.xls"
output_middle_path = "D://2//fenceng//3//jieguo//3middleval1.xls"
output_high_path = "D://2//fenceng//3//jieguo//3highval1.xls"

# 读取数据文件
low_data = pd.read_excel(low_file_path)
middle_data = pd.read_excel(middle_file_path)
high_data = pd.read_excel(high_file_path)
ml_data = pd.read_excel(ml_file_path)

# 定义函数，根据id匹配ml.xls数据，并将列名 val_Predicted_Class 改为 ml
def merge_and_output(data, ml_data, output_path):
    # 合并数据，根据id匹配ml.xls中的id, 0, 1, 2, 3, t列
    merged = pd.merge(data, ml_data[['id', 0, 1, 2, 3, 't']], on='id', how='left')

    # 将列名 val_Predicted_Class 改为 ml
    merged = merged.rename(columns={'val_Predicted_Class': 'ml'})

    # 保留所需列：id, 0, 1, 2, 3, t, ml
    result = merged[['id', 0, 1, 2, 3, 't', 'ml']]

    # 保存结果到新的xls文件
    result.to_excel(output_path, index=False)
    print(f"文件已保存到 {output_path}")

# 分别处理3lowval.xls、3middleval.xls和3highval.xls文件
merge_and_output(low_data, ml_data, output_low_path)
merge_and_output(middle_data, ml_data, output_middle_path)
merge_and_output(high_data, ml_data, output_high_path)