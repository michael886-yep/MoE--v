# 导入必要的库
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 确保 RevIN 模块路径正确（根据实际目录调整）
from layers.Invertible import RevIN  # 假设 RevIN 类在同级目录的 layers/Invertible.py 中
from layers.dctnet import dct_channel_block, dct

# ========================
# 设置随机种子
# ========================
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# ========================
# 数据加载与预处理
# ========================
# 读取 CSV 数据
data = pd.read_csv("D://vocation//xls//zuizhong//xu144-144.csv")

# 提取特征列和目标列
features = data[["shaoguan", "lianzhou", "fogang", "wl1"]]  # 特征列
target = data["wl"]  # 目标列

# 将数据分为训练集和测试集
train_size = int(len(features) * 0.8)
train_features, test_features = features[:train_size], features[train_size:]
train_target, test_target = target[:train_size], target[train_size:]

# 数据标准化（特征和目标分开处理）
scaler_features = StandardScaler()
train_features_scaled = scaler_features.fit_transform(train_features)
test_features_scaled = scaler_features.transform(test_features)

scaler_target = StandardScaler()
train_target_scaled = scaler_target.fit_transform(train_target.values.reshape(-1, 1))
test_target_scaled = scaler_target.transform(test_target.values.reshape(-1, 1))


# ========================
# 自定义数据生成函数
# ========================
def create_dataset(features, target, look_back=144, look_forward=1):
    dataX, dataY = [], []
    for i in range(len(features) - look_back - look_forward + 1):
        a = features[i:(i + look_back)]  # (144, 4)
        dataX.append(a)
        b = target[(i + look_back):(i + look_back + look_forward)]
        dataY.append(b)
    return np.array(dataX), np.array(dataY)


look_back = 144
look_forward = 1

# 生成训练集和测试集
trainX, trainY = create_dataset(train_features_scaled, train_target_scaled, look_back, look_forward)
testX, testY = create_dataset(test_features_scaled, test_target_scaled, look_back, look_forward)

# 调整数据维度为 (batch_size, seq_len, input_size)
trainX = torch.tensor(trainX, dtype=torch.float32)  # (samples, 144, 4)
trainY = torch.tensor(trainY, dtype=torch.float32).view(-1, look_forward)  # (samples, 1)

testX = torch.tensor(testX, dtype=torch.float32)
testY = torch.tensor(testY, dtype=torch.float32).view(-1, look_forward)


# ========================
# 定义 LSTM-Transformer 模型（集成 RevIN）
# ========================
class LSTMTransformerPredictor(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, lstm_layers=2,
                 d_model=64, nhead=4, transformer_layers=2, seq_length=144, rev_enabled=True):
        super().__init__()

        # RevIN 层（输入归一化）
        self.rev = RevIN(num_features=input_size, affine=True) if rev_enabled else None

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            # dropout=0.2
        )

        # 特征维度转换（LSTM 输出 -> Transformer 输入）
        self.transform = nn.Linear(hidden_size, d_model)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            # dropout=0.2,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )
        self.linear_transform = nn.Linear(d_model, input_size)  #

        # 输出层
        #self.decoder = nn.Linear(d_model, 1)
        self.decoder = nn.Linear(input_size,1)
    def forward(self, x):
        # RevIN 归一化（输入数据）
        if self.rev is not None:
            x = self.rev(x, mode='norm')  # 训练时自动记录统计量

        # LSTM处理 (batch_size, 144, 4) -> (batch_size, 144, 64)
        lstm_out, _ = self.lstm(x)

        # 特征转换 (batch_size, 144, 64) -> (batch_size, 144, 64)
        transformed = self.transform(lstm_out)

        # Transformer处理
        transformer_out = self.transformer_encoder(transformed)
        out1 = self.linear_transform(transformer_out)  #
        #(32,144,64)
        if self.rev is not None:
            transformer_out2 = self.rev(out1, mode='denorm')

        # 取最后一个时间步预测 (batch_size, 64)
        # last_time_step_output = transformer_out2[:, -1, :]
        last_time_step_output = out1[:, -1, :]
        #(32,64)
        # 输出预测 (batch_size, 1)
        predictions = self.decoder(last_time_step_output)

        # RevIN 反归一化（输出数据）
        #if self.rev is not None:
            #predictions = self.rev(predictions, mode='denorm')

        return predictions


# ========================
# 模型初始化与参数设置
# ========================
model = LSTMTransformerPredictor(
    input_size=4,
    hidden_size=8,
    lstm_layers=1,
    d_model=16,
    nhead=16,
    transformer_layers=1,
    rev_enabled=False  # 启用 RevIN
)

# ========================
# 数据加载器
# ========================
batch_size = 32
train_dataset = TensorDataset(trainX, trainY)
test_dataset = TensorDataset(testX, testY)


# 自定义 collate_fn 确保数据维度正确
def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    return inputs, targets


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ========================
# 训练配置
# ========================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 1000
patience = 20
min_loss = np.inf
patience_counter = 0

train_losses = []
val_losses = []

# ========================
# 训练循环
# ========================
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(test_loader.dataset)
    val_losses.append(val_loss)

    # 打印损失
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # 早停机制
    if val_loss < min_loss:
        min_loss = val_loss
        patience_counter = 0
        best_model = model.state_dict()
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping")
        break

# 加载最佳模型
model.load_state_dict(best_model)

# ========================
# 预测与评估
# ========================
model.eval()
with torch.no_grad():
    testPredict = model(testX).numpy()
    trainPredict = model(trainX).numpy()

print(f"trainPredict shape: {trainPredict.shape}")
print(f"testPredict shape: {testPredict.shape}")

# 反标准化处理（目标变量使用独立的 StandardScaler）
testPredict = scaler_target.inverse_transform(testPredict).flatten()
testY = scaler_target.inverse_transform(testY).flatten()
trainPredict = scaler_target.inverse_transform(trainPredict).flatten()
trainY = scaler_target.inverse_transform(trainY).flatten()

# 打印评估指标
print(f"Train RMSE: {mean_squared_error(trainY, trainPredict, squared=False):.4f}")
print(f"Test RMSE:  {mean_squared_error(testY, testPredict, squared=False):.4f}")
print(f"Train MAE:  {mean_absolute_error(trainY, trainPredict):.4f}")
print(f"Test MAE:   {mean_absolute_error(testY, testPredict):.4f}")
print(f"Train R²:   {r2_score(trainY, trainPredict):.4f}")
print(f"Test R²:    {r2_score(testY, testPredict):.4f}")

# ========================
# 保存预测结果
# ========================
pd.DataFrame({'Actual': testY, 'Predicted': testPredict}).to_excel(
    "D://vocation//xls//daochu//test_results.xlsx", index=False
)
pd.DataFrame({'Actual1': trainY, 'Predicted1': trainPredict}).to_excel(
    "D://vocation//xls//daochu//train_results.xlsx", index=False
)

# ========================
# 绘制预测结果对比图
# ========================
plt.figure(figsize=(18, 6))

# 绘制训练集结果
plt.subplot(1, 2, 1)
plt.plot(trainY, label='Actual (Train)', alpha=0.7)
plt.plot(trainPredict, label='Predicted (Train)', alpha=0.7)
plt.title('Training Set: Actual vs Predicted')
plt.xlabel('Time Steps')
plt.ylabel('Water Level (wl)')
plt.legend()
plt.grid(True)

# 绘制测试集结果
plt.subplot(1, 2, 2)
plt.plot(testY, label='Actual (Test)', alpha=0.7)
plt.plot(testPredict, label='Predicted (Test)', alpha=0.7)
plt.title('Test Set: Actual vs Predicted')
plt.xlabel('Time Steps')
plt.ylabel('Water Level (wl)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()