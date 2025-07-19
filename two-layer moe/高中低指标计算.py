import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd

# 文件路径列表
input_files = [
    {"path": "D://2//fenceng//3//jieguo//3lowval2.xls", "name": "3lowval2"},
    {"path": "D://2//fenceng//3//jieguo//3middleval2.xls", "name": "3middleval2"},
    {"path": "D://2//fenceng//3//jieguo//3highval2.xls", "name": "3highval2"}
]

# 定义计算指标的函数
def calculate_metrics(file_path, file_name):
    # 读取数据
    df = pd.read_excel(file_path)
    x = df['p']
    y = df['t']

    # 计算指标
    RMSE = np.sqrt(mean_squared_error(y, x))
    R2 = r2_score(y, x)
    R, _ = pearsonr(x, y)
    MAE = mean_absolute_error(y, x)

    # 输出结果
    print(f"文件: {file_name}")
    print(f"R: {R}")
    # print(f"R2: {R2}")
    print(f"MAE: {MAE}")
    print(f"RMSE: {RMSE}")
    print("-" * 50)

# 批量处理文件
for file in input_files:
    calculate_metrics(file["path"], file["name"])
