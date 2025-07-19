import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder  # 添加 LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier  # 导入XGBoost模型

# 设置随机种子
seed = 42
np.random.seed(seed)

# 读取CSV数据（已具有列名，date为时间列，ml为目标列，f1~f576为特征列）
data = pd.read_csv("D://2//dongmoe2//3//canchafenlei//3dong2can.csv")  # 替换为您的CSV文件路径

# 将时间列解析为日期时间格式
data['date'] = pd.to_datetime(data['date'], format='%Y/%m/%d')

# 按时间划分数据集
train_data = data[(data['date'] >= '2020-01-01') & (data['date'] <= '2038-06-14')]
val_data = data[(data['date'] >= '2038-06-15') & (data['date'] <= '2042-10-09')]

# 提取训练集和验证集的目标列和特征列（剔除时间列）
train_target = train_data['class']
train_features = train_data.drop(columns=['class', 'date'])

val_target = val_data['class']
val_features = val_data.drop(columns=['class', 'date'])

# 对目标变量进行标签编码（从1, 2, 3转换为0, 1, 2）
label_encoder = LabelEncoder()  # 创建 LabelEncoder 实例
train_target_encoded = label_encoder.fit_transform(train_target)  # 将训练集目标变量转换为整数标签
val_target_encoded = label_encoder.transform(val_target)  # 将验证集目标变量转换为整数标签

# 获取类别列表（已编码）
class_labels = label_encoder.classes_  # 获取原始类别标签对应的顺序

# 数据标准化 - 仅在训练集上拟合
scaler_features = StandardScaler()
train_features_scaled = scaler_features.fit_transform(train_features)

# 在验证集上使用相同的scaler进行标准化
val_features_scaled = scaler_features.transform(val_features)

# 创建XGBoost模型
model = XGBClassifier(n_estimators=15, random_state=seed, use_label_encoder=False, eval_metric='logloss')
model.fit(train_features_scaled, train_target_encoded)

# 预测训练集和验证集
trainPredict = model.predict(train_features_scaled)
valPredict = model.predict(val_features_scaled)

# 将预测结果转换回原始标签
trainPredict_decoded = label_encoder.inverse_transform(trainPredict)
valPredict_decoded = label_encoder.inverse_transform(valPredict)

# 输出结果
print("训练集指标：")
print("Accuracy:", accuracy_score(train_target, trainPredict_decoded))
print("Precision:", precision_score(train_target, trainPredict_decoded, average='macro'))
print("Recall:", recall_score(train_target, trainPredict_decoded, average='macro'))
print("F1 Score:", f1_score(train_target, trainPredict_decoded, average='macro'))

print("\n验证集指标：")
print("Accuracy:", accuracy_score(val_target, valPredict_decoded))
print("Precision:", precision_score(val_target, valPredict_decoded, average='macro'))
print("Recall:", recall_score(val_target, valPredict_decoded, average='macro'))
print("F1 Score:", f1_score(val_target, valPredict_decoded, average='macro'))

# 计算混淆矩阵
train_conf_matrix = confusion_matrix(train_target, trainPredict_decoded, labels=class_labels)
val_conf_matrix = confusion_matrix(val_target, valPredict_decoded, labels=class_labels)

# 输出混淆矩阵（注明类别）
print("\n训练集混淆矩阵：")
print(pd.DataFrame(train_conf_matrix, index=[f"True_{label}" for label in class_labels],
                   columns=[f"Pred_{label}" for label in class_labels]))

print("\n验证集混淆矩阵：")
print(pd.DataFrame(val_conf_matrix, index=[f"True_{label}" for label in class_labels],
                   columns=[f"Pred_{label}" for label in class_labels]))

# 找到每个样本预测的类别
val_predictions = pd.DataFrame(valPredict_decoded, columns=["class"])  # val_Predicted_Class
val_predictions.index = range(len(val_predictions))  # 设置索引为顺序索引

# 保存验证集预测结果
val_predictions.to_excel("D://2//dongmoe2//3//canchafenlei//3canval.xls", index=False)

print("验证集预测结果已保存为 'val_classes.xls'")