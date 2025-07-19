import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# 设置随机种子
seed = 42
np.random.seed(seed)

# 读取CSV数据（已具有列名，id为样本标识符列，ml为目标列，f1~f576为特征列）
data = pd.read_csv("D://2//fenceng//12//new//12low.csv")  # 替换为您的CSV文件路径

# 按 id 列划分数据集
train_data = data[(data['id'] >= 1) & (data['id'] <= 6654)]
val_data = data[(data['id'] >= 6655) & (data['id'] <= 8210)]

# 提取训练集和验证集的目标列和特征列（保留 id 列以便输出）
train_target = train_data['ml']
train_features = train_data.drop(columns=['ml'])

val_target = val_data['ml']
val_features = val_data.drop(columns=['ml'])

# 获取类别列表
class_labels = sorted(train_target.unique())

# 数据标准化 - 仅在训练集上拟合
scaler_features = StandardScaler()
train_features_scaled = scaler_features.fit_transform(train_features.drop(columns=['id']))

# 在验证集上使用相同的scaler进行标准化
val_features_scaled = scaler_features.transform(val_features.drop(columns=['id']))

# 创建随机森林模型
model = RandomForestClassifier(n_estimators=2, random_state=seed)
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

# 找到每个样本预测的类别，并保留对应的 id 列
val_predictions = pd.DataFrame({
    "id": val_features['id'],
    "val_Predicted_Class": valPredict
})

# 保存验证集预测结果
val_predictions.to_excel("D://2//fenceng//12//jieguo//12lowval.xls", index=False)

print("验证集预测结果已保存为 '3jingval.xls'")