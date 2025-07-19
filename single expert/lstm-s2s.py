import os  # 用于处理文件路径
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# 固定随机种子
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# 读取 CSV 数据
data = pd.read_csv("D://vocation//xls//zuizhong//xu144-144.csv")

# 提取特征列和目标列
features = data[["shaoguan", "lianzhou", "fogang", "wl1"]]  # 特征列
target = data["wl"]  # 目标列

# 将数据分为训练集和测试集
train_size = int(len(features) * 0.8)
train_features, test_features = features[:train_size], features[train_size:]
train_target, test_target = target[:train_size], target[train_size:]

# 数据标准化 - 仅对训练集拟合
scaler_features = StandardScaler()
train_features_scaled = scaler_features.fit_transform(train_features)
test_features_scaled = scaler_features.transform(test_features)

# 对目标变量进行标准化 - 仅对训练集拟合
scaler_target = StandardScaler()
train_target_scaled = scaler_target.fit_transform(train_target.values.reshape(-1, 1))
test_target_scaled = scaler_target.transform(test_target.values.reshape(-1, 1))

# 创建数据集
def create_dataset(features, target, look_back=1, look_forward=3):
    dataX, dataY = [], []
    for i in range(len(features) - look_back - look_forward + 1):
        a = features[i:(i + look_back)]
        dataX.append(a)
        b = target[(i + look_back):(i + look_back + look_forward)]
        dataY.append(b)
    return np.array(dataX), np.array(dataY)

look_back = 144
look_forward = 1

trainX, trainY = create_dataset(train_features_scaled, train_target_scaled, look_back, look_forward)
testX, testY = create_dataset(test_features_scaled, test_target_scaled, look_back, look_forward)

# 构建 LSTM 模型
units = 13
repeat = look_forward
regular = 0.001

model = Sequential()
model.add(LSTM(units, activation='relu', input_shape=(look_back, trainX.shape[2])))
model.add(RepeatVector(repeat))
model.add(LSTM(units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(regular)))
model.add(TimeDistributed(Dense(1)))

custom_adam = Adam(learning_rate=0.01, clipvalue=1.0)
model.compile(loss='mean_squared_error', optimizer=custom_adam)

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# 训练模型
history = model.fit(trainX, trainY, epochs=1000, batch_size=32, validation_data=(testX, testY), verbose=2, callbacks=[early_stopping])
model.summary()

# 预测结果
testPredict = model.predict(testX)
trainPredict = model.predict(trainX)

# 反标准化预测结果，调整维度以进行评估
testPredict = scaler_target.inverse_transform(testPredict[:, -1, 0].reshape(-1, 1)).flatten()
testY = scaler_target.inverse_transform(testY[:, -1].reshape(-1, 1)).flatten()
trainPredict = scaler_target.inverse_transform(trainPredict[:, -1, 0].reshape(-1, 1)).flatten()
trainY = scaler_target.inverse_transform(trainY[:, -1].reshape(-1, 1)).flatten()

# 计算评估指标
test_rmse = mean_squared_error(testY, testPredict, squared=False)
train_rmse = mean_squared_error(trainY, trainPredict, squared=False)

train_r2 = r2_score(trainY, trainPredict)
test_r2 = r2_score(testY, testPredict)

train_mae = mean_absolute_error(trainY, trainPredict)
test_mae = mean_absolute_error(testY, testPredict)

# 输出指标
print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
print("Train R²:", train_r2)
print("Test R²:", test_r2)
print("Train MAE:", train_mae)
print("Test MAE:", test_mae)

# 绘制损失曲线
plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='test_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()

# 绘制预测对比图
plt.figure()
plt.plot(testY, label='Observed (144th day)')
plt.plot(testPredict, label='Predicted (144th day)')
plt.xlabel('Index')
plt.ylabel('wl')
plt.title('144 Days Ahead Prediction Comparison')
plt.legend()
plt.show()

# 创建DataFrame保存预测结果
# 对于训练集
train_results = pd.DataFrame({
    't': trainY,
    'p': trainPredict
})

# 对于测试集
test_results = pd.DataFrame({
    't': testY,
    'p': testPredict
})

# 将结果保存为Excel文件
train_results.to_excel("D://2//results//12//lstmed//train.xls", index=False)
test_results.to_excel("D://2//results//12//lstmed//test.xls", index=False)