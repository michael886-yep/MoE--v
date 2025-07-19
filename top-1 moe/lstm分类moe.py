import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
# 设置随机种子
seed = 42
tf.random.set_seed(seed)

# 读取CSV数据
data = pd.read_csv("D://2//jingmoe//3//fen//3jingtaimoe.csv")  # 替换为您的CSV文件路径

# 将日期列解析为日期时间格式
data['date'] = pd.to_datetime(data['date'], format='%Y/%m/%d')
data.set_index('date', inplace=True)

# 按时间划分训练集和验证集
train_data = data.loc['2020/1/1':'2038/6/14']
val_data = data.loc['2038/6/15':'2042/10/9']

# 提取训练集和验证集的目标列和特征列
train_target = train_data['ml']
train_features = train_data.drop(columns=['ml'])

val_target = val_data['ml']
val_features = val_data.drop(columns=['ml'])

# 数据标准化 - 仅在训练集上拟合
scaler_features = StandardScaler()
train_features_scaled = scaler_features.fit_transform(train_features)
val_features_scaled = scaler_features.transform(val_features)

# 将目标列进行 One-Hot 编码
train_target_encoded = tf.keras.utils.to_categorical(train_target, num_classes=4)
val_target_encoded = tf.keras.utils.to_categorical(val_target, num_classes=4)

# 创建数据集
def create_dataset(features, target, look_back=1):
    dataX, dataY = [], []
    for i in range(len(features)):
        a = features[i:i + look_back]  # 包含了从索引 i 到 i + look_back - 1 的所有行
        dataX.append(a)
        dataY.append(target[i])
    return np.array(dataX), np.array(dataY)

look_back = 1
trainX, trainY = create_dataset(train_features_scaled, train_target_encoded, look_back)
valX, valY = create_dataset(val_features_scaled, val_target_encoded, look_back)

# 创建LSTM模型
model = Sequential()
model.add(LSTM(64, input_shape=(look_back, train_features.shape[1])))  # 输入维度根据特征数动态调整
model.add(Dense(4, activation="softmax"))


# 设置自定义学习率
custom_learning_rate = 0.00001  # 将学习率设置为 0.0001
optimizer = Adam(learning_rate=custom_learning_rate)

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
model.summary()

#model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])
#model.summary()

# 提前停止回调
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# 训练模型
history = model.fit(trainX, trainY, epochs=200, batch_size=32, validation_data=(valX, valY), verbose=2, callbacks=[early_stopping])

# 预测验证集
valPredict = model.predict(valX)
trainPredict = model.predict(trainX)

# 计算训练集指标
train_conf_matrix = confusion_matrix(np.argmax(trainY, axis=1), np.argmax(trainPredict, axis=1))
train_accuracy = accuracy_score(np.argmax(trainY, axis=1), np.argmax(trainPredict, axis=1))
train_recall = recall_score(np.argmax(trainY, axis=1), np.argmax(trainPredict, axis=1), average='macro')
train_precision = precision_score(np.argmax(trainY, axis=1), np.argmax(trainPredict, axis=1), average='macro')
train_f1 = f1_score(np.argmax(trainY, axis=1), np.argmax(trainPredict, axis=1), average='macro')

# 计算验证集指标
val_conf_matrix = confusion_matrix(np.argmax(valY, axis=1), np.argmax(valPredict, axis=1))
val_accuracy = accuracy_score(np.argmax(valY, axis=1), np.argmax(valPredict, axis=1))
val_recall = recall_score(np.argmax(valY, axis=1), np.argmax(valPredict, axis=1), average='macro')
val_precision = precision_score(np.argmax(valY, axis=1), np.argmax(valPredict, axis=1), average='macro')
val_f1 = f1_score(np.argmax(valY, axis=1), np.argmax(valPredict, axis=1), average='macro')

# 输出指标
print("训练集混淆矩阵：", train_conf_matrix)
print("训练集准确率：", train_accuracy)
print("训练集召回率", train_recall)
print("训练集精确率：", train_precision)
print("训练集F1分数：", train_f1)

print("验证集混淆矩阵：", val_conf_matrix)
print("验证集准确率：", val_accuracy)
print("验证集召回率", val_recall)
print("验证集精确率：", val_precision)
print("验证集F1分数：", val_f1)

# 输出验证集预测结果
val_predicted_classes = np.argmax(valPredict, axis=1)
val_predictions = pd.DataFrame(val_predicted_classes, columns=["val_Predicted_Class"])
val_predictions.index = val_data.index
val_predictions.to_excel("D://2//jingmoe//3//fen//3jingval.xls", index=False)  # 替换为实际输出路径

# 输出训练集预测结果
#train_predicted_classes = np.argmax(trainPredict, axis=1)
#train_predictions = pd.DataFrame(train_predicted_classes, columns=["train_Predicted_Class"])
#train_predictions.index = train_data.index
#train_predictions.to_excel("path_to_train_predictions.xlsx", index=True)  # 替换为实际输出路径