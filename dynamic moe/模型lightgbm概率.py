import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier  # 导入LightGBM模型

# 设置随机种子
seed = 42
np.random.seed(seed)

# 读取CSV数据（已具有列名，date为时间列，ml为目标列，f1~f576为特征列）
data = pd.read_csv("D://2//dongmoe2//12//canchafenlei//12dong1moe.csv")  # 替换为您的CSV文件路径

# 将时间列解析为日期时间格式
data['date'] = pd.to_datetime(data['date'], format='%Y/%m/%d')

# 按时间划分数据集
train_data = data[(data['date'] >= '2020-01-01') & (data['date'] <= '2038-03-20')]
val_data = data[(data['date'] >= '2038-03-21') & (data['date'] <= '2042-06-23')]

# 提取训练集和验证集的目标列和特征列（剔除时间列）
train_target = train_data['ml']
train_features = train_data.drop(columns=['ml', 'date'])

val_target = val_data['ml']
val_features = val_data.drop(columns=['ml', 'date'])

# 获取类别列表
class_labels = sorted(train_target.unique())

# 数据标准化 - 仅在训练集上拟合
scaler_features = StandardScaler()
train_features_scaled = scaler_features.fit_transform(train_features)

# 在验证集上使用相同的scaler进行标准化
val_features_scaled = scaler_features.transform(val_features)

# 创建LightGBM模型
model = LGBMClassifier(n_estimators=70, random_state=seed, learning_rate=0.1)
model.fit(train_features_scaled, train_target)

# 预测训练集和验证集
trainPredict = model.predict(train_features_scaled)
valPredict = model.predict(val_features_scaled)

# 输出结果
print("训练集指标：")
print("Accuracy:", accuracy_score(train_target, trainPredict))
print("Precision:", precision_score(train_target, trainPredict, average='macro'))
print("Recall:", recall_score(train_target, trainPredict, average='macro'))
print("F1 Score:", f1_score(train_target, trainPredict, average='macro'))

print("\n验证集指标：")
print("Accuracy:", accuracy_score(val_target, valPredict))
print("Precision:", precision_score(val_target, valPredict, average='macro'))
print("Recall:", recall_score(val_target, valPredict, average='macro'))
print("F1 Score:", f1_score(val_target, valPredict, average='macro'))

# 计算混淆矩阵
train_conf_matrix = confusion_matrix(train_target, trainPredict, labels=class_labels)
val_conf_matrix = confusion_matrix(val_target, valPredict, labels=class_labels)

# 输出混淆矩阵（注明类别）
print("\n训练集混淆矩阵：")
print(pd.DataFrame(train_conf_matrix, index=[f"True_{label}" for label in class_labels],
                   columns=[f"Pred_{label}" for label in class_labels]))

print("\n验证集混淆矩阵：")
print(pd.DataFrame(val_conf_matrix, index=[f"True_{label}" for label in class_labels],
                   columns=[f"Pred_{label}" for label in class_labels]))

# 找到验证集每一类的概率
valPredict_proba = model.predict_proba(val_features_scaled)
probability_columns = [f"Prob_Class_{label}" for label in class_labels]

val_probabilities = pd.DataFrame(valPredict_proba, columns=probability_columns)
val_probabilities.index = range(len(val_probabilities))  # 设置索引为顺序索引

# 将预测类别与概率合并
val_predictions = pd.DataFrame(valPredict, columns=["val_Predicted_Class"])  # val_Predicted_Class
val_results = pd.concat([val_predictions, val_probabilities], axis=1)

# 保存验证集预测结果和概率/
val_results.to_excel("D://2//dongmoe2//12//canchafenlei//12probabilities.xls", index=False)

print("验证集预测结果及概率已保存为 '3jingval_with_probabilities.xls'")