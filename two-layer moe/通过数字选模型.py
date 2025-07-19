import pandas as pd

# 文件路径
input_files = [
    {"input": "D://2//fenceng//3//jieguo//3lowval1.xls", "output": "D://2//fenceng//3//jieguo//3lowval2.xls"},
    {"input": "D://2//fenceng//3//jieguo//3middleval1.xls", "output": "D://2//fenceng//3//jieguo//3middleval2.xls"},
    {"input": "D://2//fenceng//3//jieguo//3highval1.xls", "output": "D://2//fenceng//3//jieguo//3highval2.xls"}
]

# 定义处理函数
def process_file(input_path, output_path):
    # 读取原始 Excel 文件
    data = pd.read_excel(input_path)

    # 确保列名是字符串类型
    data.columns = data.columns.astype(str)

    # 使用 ml 列的值动态选择对应列的值，并创建新列 p
    # 将 ml 的值转换为整数后再转为字符串，确保与列名匹配
    data['p'] = data.apply(lambda row: row[str(int(row['ml']))], axis=1)

    # 选择新列 p、t 和 id 列
    output_data = data[['id', 'p', 't']]

    # 输出为新的 Excel 文件
    output_data.to_excel(output_path, index=False)
    print(f"处理完成，文件已保存为 {output_path}")

# 批量处理文件
for file in input_files:
    process_file(file["input"], file["output"])