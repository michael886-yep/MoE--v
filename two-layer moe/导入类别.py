import pandas as pd

# 文件路径
low_file_path = "D://2//fenceng//12//new//12low.xlsx"  # 替换为3low.xlsx的实际路径
middle_file_path = "D://2//fenceng//12//new//12middle.xlsx"  # 替换为3middle.xlsx的实际路径
high_file_path = "D://2//fenceng//12//new//12high.xlsx"  # 替换为3high.xlsx的实际路径
cc_file_path = "D://2//fenceng//12//new//cc.xls"  # 替换为cc.xls的实际路径

# 输出文件路径
output_low_path = "D://2//fenceng//12//new//12low.csv"
output_middle_path = "D://2//fenceng//12//new//12middle.csv"
output_high_path = "D://2//fenceng//12//new//12high.csv"

# 读取数据文件，指定引擎为 openpyxl
low_data = pd.read_excel(low_file_path, engine='openpyxl')
middle_data = pd.read_excel(middle_file_path, engine='openpyxl')
high_data = pd.read_excel(high_file_path, engine='openpyxl')
cc_data = pd.read_excel(cc_file_path)

# 定义函数，根据id匹配ml列并替代Predicted_Class列
def replace_predicted_class(data, cc_data, output_path):
    # 合并数据，根据id匹配ml列
    merged = pd.merge(data, cc_data, on='id', how='left')

    # 用ml列的值替代Predicted_Class列的值，并将列名改为ml
    merged['ml'] = merged['ml']
    merged = merged.drop(columns=['Predicted_Class'])

    # 保存结果到CSV文件，避免行间距问题
    merged.to_csv(output_path, index=False, line_terminator='\n')
    print(f"文件已保存到 {output_path}")

# 分别处理3low.xlsx、3middle.xlsx和3high.xlsx文件
replace_predicted_class(low_data, cc_data, output_low_path)
replace_predicted_class(middle_data, cc_data, output_middle_path)
replace_predicted_class(high_data, cc_data, output_high_path)