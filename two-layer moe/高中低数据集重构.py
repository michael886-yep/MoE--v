import pandas as pd

# 文件路径
data_file_path = "D://2//fenceng//12//runoff//144data.csv"  # 替换为36data.csv的实际路径
train_file_path = "D://2//fenceng//12//runoff//train.xls"  # 替换为train.xls的实际路径
test_file_path = "D://2//fenceng//12//runoff//test.xls"  # 替换为test.xls的实际路径

# 输出文件路径
output_low_path = "D://2//fenceng//12//new//12low.xlsx"
output_middle_path = "D://2//fenceng//12//new//12middle.xlsx"
output_high_path = "D://2//fenceng//12//new//12high.xlsx"

# 读取数据文件
data = pd.read_csv(data_file_path)
train = pd.read_excel(train_file_path)
test = pd.read_excel(test_file_path)

# 合并train和test数据
combined = pd.concat([train, test], ignore_index=True)

# 定义函数，根据Predicted_Class筛选并匹配数据
def filter_and_merge(predicted_class, output_path):
    # 筛选出指定Predicted_Class的数据
    filtered = combined[combined['Predicted_Class'] == predicted_class]

    # 根据id匹配36data.csv中的f1~f576列
    merged = pd.merge(filtered, data, on='id')

    # 只保留id, Predicted_Class, f1~f576列
    result = merged[['id', 'Predicted_Class'] + [col for col in merged.columns if col.startswith('f')]]

    # 输出到文件，使用 openpyxl 引擎
    result.to_excel(output_path, index=False, engine='openpyxl')
    print(f"文件已保存到 {output_path}")

# 分别处理Predicted_Class为1、2、3的情况
filter_and_merge(predicted_class=1, output_path=output_low_path)
filter_and_merge(predicted_class=2, output_path=output_middle_path)
filter_and_merge(predicted_class=3, output_path=output_high_path)