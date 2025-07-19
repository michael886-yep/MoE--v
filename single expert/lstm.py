import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)   # 数据集变更后的序列lstm

# 读取 CSV 数据
data = pd.read_csv("D://vocation//xls//zuizhong//xu144-144.csv")

# 提取特征列和目标列
features = data[["shaoguan", "lianzhou", "fogang","wl1"]]  # 特征列 ,"fgtem","lztem","sgtem"
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
model = Sequential()
model.add(LSTM(20, input_shape=(look_back, 4)))  # 添加L2正则化
# model.add(Dropout(0.3))
model.add(Dense(1))
# custom_rmsprop = RMSprop(learning_rate=0.00001, clipvalue=1.0)
custom_adam = Adam(learning_rate=0.00001, clipvalue=1.0)
model.compile(loss='mean_squared_error', optimizer=custom_adam)
# 配置SGD优化器
# custom_sgd = SGD(learning_rate=0.0001, momentum=0.9, clipvalue=1.0)
# model.compile(loss='mean_squared_error', optimizer=custom_sgd)
# model.compile(loss='mean_squared_error', optimizer=custom_rmsprop)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# 训练模型
history = model.fit(trainX, trainY, epochs=1000, batch_size=32, validation_data=(testX, testY), verbose=2, callbacks=[early_stopping])
# history = model.fit(trainX, trainY, epochs=200, batch_size=32, validation_data=(testX, testY), verbose=2)
model.summary()
# 绘制训练和验证损失曲线
plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='test_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('loss')
plt.legend()
plt.show()

# 预测结果
testPredict = model.predict(testX)
trainPredict = model.predict(trainX)

# 反标准化预测结果，调整维度以进行评估
testPredict = scaler_target.inverse_transform(testPredict).flatten()
testY = scaler_target.inverse_transform(testY).flatten()
trainPredict = scaler_target.inverse_transform(trainPredict).flatten()
trainY = scaler_target.inverse_transform(trainY).flatten()

# 计算 RMSE
test_rmse = mean_squared_error(testY, testPredict, squared=False)
train_rmse = mean_squared_error(trainY, trainPredict, squared=False)
test_mae = mean_absolute_error(testY, testPredict)
train_mae = mean_absolute_error(trainY, trainPredict)
print("Train RMSE for last forecast day:", train_rmse)
print("Test RMSE for last forecast day:", test_rmse)
print("Train MAE for last forecast day:", train_mae)
print("Test MAE for last forecast day:", test_mae)
# 计算 R2
train_r2 = r2_score(trainY, trainPredict)
test_r2 = r2_score(testY, testPredict)
print("Train R2 for last forecast day:", train_r2)
print("Test R2 for last forecast day:", test_r2)

# 绘制验证集的实际值和预测值，针对最后一个预测点
plt.figure()
plt.plot(testY, label='observed')
plt.plot(testPredict, label='predicted')
plt.xlabel('index')
plt.ylabel('wl')
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
train_results.to_excel("D://2//results//12//lstm//train.xls", index=False)
test_results.to_excel("D://2//results//12//lstm//test.xls", index=False)