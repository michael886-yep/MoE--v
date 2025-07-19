import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

# 设置随机种子
seed = 42
np.random.seed(seed)

# 读取CSV数据（具有列名，id列为唯一标识符，date列为日期，class列为目标列，f1~f576为特征列）
data = pd.read_csv("D://2//fenceng//12//runoff//144data.csv")  # 替换为您的CSV文件路径

# 将日期列解析为日期时间格式
data['date'] = pd.to_datetime(data['date'], format='%Y/%m/%d')

# 按日期划分训练集和测试集
train_data = data[(data['date'] >= '2020-01-01') & (data['date'] <= '2038-03-20')]
test_data = data[(data['date'] >= '2038-03-21') & (data['date'] <= '2042-06-23')]

# 提取训练集和测试集的目标列和特征列
train_target = train_data['class']
train_features = train_data.filter(regex='^f\\d+$')  # 提取f1到f576的特征列

test_target = test_data['class']
test_features = test_data.filter(regex='^f\\d+$')  # 提取f1到f576的特征列

# 数据标准化 - 仅在训练集上拟合
scaler_features = StandardScaler()
train_features_scaled = scaler_features.fit_transform(train_features)
test_features_scaled = scaler_features.transform(test_features)

# 创建XGBoost模型
model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=seed)  # 设置学习率
model.fit(train_features_scaled, train_target)

# 预测训练集和测试集
trainPredict = model.predict(train_features_scaled)
testPredict = model.predict(test_features_scaled)

# 输出结果
print("训练集指标：")
print("Accuracy:", accuracy_score(train_target, trainPredict))
print("Precision:", precision_score(train_target, trainPredict, average='macro'))
print("Recall:", recall_score(train_target, trainPredict, average='macro'))
print("F1 Score:", f1_score(train_target, trainPredict, average='macro'))

print("\n测试集指标：")
print("Accuracy:", accuracy_score(test_target, testPredict))
print("Precision:", precision_score(test_target, testPredict, average='macro'))
print("Recall:", recall_score(test_target, testPredict, average='macro'))
print("F1 Score:", f1_score(test_target, testPredict, average='macro'))

# 计算混淆矩阵
class_labels = sorted(train_target.unique())
train_conf_matrix = confusion_matrix(train_target, trainPredict, labels=class_labels)
test_conf_matrix = confusion_matrix(test_target, testPredict, labels=class_labels)

# 输出混淆矩阵（注明类别）
print("\n训练集混淆矩阵：")
print(pd.DataFrame(train_conf_matrix, index=[f"True_{label}" for label in class_labels],
                   columns=[f"Pred_{label}" for label in class_labels]))

print("\n测试集混淆矩阵：")
print(pd.DataFrame(test_conf_matrix, index=[f"True_{label}" for label in class_labels],
                   columns=[f"Pred_{label}" for label in class_labels]))

# 保存训练集预测结果
train_results = train_data[['id', 'class']].copy()
train_results['Predicted_Class'] = trainPredict

# 保存测试集预测结果
test_results = test_data[['id', 'class']].copy()
test_results['Predicted_Class'] = testPredict

# 输出训练集和测试集结果到Excel
output_train_path = "D://2//fenceng//12//runoff//train.xls"
output_test_path = "D://2//fenceng//12//runoff//test.xls"

train_results.to_excel(output_train_path, index=False)
test_results.to_excel(output_test_path, index=False)

print(f"训练集预测结果已保存到 {output_train_path}")
print(f"测试集预测结果已保存到 {output_test_path}")