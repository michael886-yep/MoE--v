# Encoder
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from layers.dctnet import dct_channel_block, dct

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# 读取 CSV 数据
data = pd.read_csv("D://vocation//xls//zuizhong//xu144-36.csv")

# 提取特征列和目标列
features = data[["shaoguan", "lianzhou", "fogang", "wl1"]]  # 特征列
target = data["wl"]  # 目标列

# 将数据分为训练集和测试集
train_size = int(len(features) * 0.8)
train_features, test_features = features[:train_size], features[train_size:]
train_target, test_target = target[:train_size], target[train_size:]

# 数据标准化
scaler_features = StandardScaler()
train_features_scaled = scaler_features.fit_transform(train_features)
test_features_scaled = scaler_features.transform(test_features)

scaler_target = StandardScaler()
train_target_scaled = scaler_target.fit_transform(train_target.values.reshape(-1, 1))
test_target_scaled = scaler_target.transform(test_target.values.reshape(-1, 1))


# ========================
# 修改点1：调整数据生成函数（保持不变）
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

trainX, trainY = create_dataset(train_features_scaled, train_target_scaled, look_back, look_forward)
testX, testY = create_dataset(test_features_scaled, test_target_scaled, look_back, look_forward)

# 调整数据维度为 (batch_size, seq_len, input_size)
trainX = torch.tensor(trainX, dtype=torch.float32)  # (samples, 144, 4)
trainY = torch.tensor(trainY, dtype=torch.float32).view(-1, look_forward)  # (samples, 1)

testX = torch.tensor(testX, dtype=torch.float32)
testY = torch.tensor(testY, dtype=torch.float32).view(-1, look_forward)


# ========================
# 修改点2：定义经典 Transformer 模型
# ========================
class ClassicTransformer(nn.Module):
    def __init__(self, input_dim=4, d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()

        # 位置编码器（经典Transformer必需）
        self.positional_encoding = nn.Parameter(torch.zeros(1, look_back, d_model))

        # 输入特征投影到d_model维度
        self.input_proj = nn.Linear(input_dim, d_model)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # FECAM
        self.dct_layer = dct_channel_block(16)   # d_model
        self.dct_norm = nn.LayerNorm([16], eps=1e-6)   #d_model

        # 输出层
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        # 添加位置编码 (batch_size, 144, 4) -> (batch_size, 144, d_model)
        x = self.input_proj(x) + self.positional_encoding[:, :x.size(1), :]

        # Transformer处理
        transformer_out = self.transformer_encoder(x)  # (batch_size, 144, d_model)

        # 加入dct模块
        #mid  = self.dct_layer(transformer_out)
        #transformer_out = transformer_out+mid
        #transformer_out = self.dct_norm(transformer_out) #norm 144

        # 取最后一个时间步预测
        predictions = self.decoder(transformer_out[:, -1, :])
        return predictions


# ========================
# 初始化模型（关键参数）
# ========================
model = ClassicTransformer(
    input_dim=4,  # 输入特征维度（4个气象站数据）
    d_model=8,  # 特征维度（与Transformer的d_model一致）
    nhead=8,  # 多头注意力头数
    num_encoder_layers=1,  # Transformer编码器层数
    dim_feedforward=32,  # 前馈网络隐藏层维度
    dropout=0.1  # Dropout概率
)

# ========================
# 调整数据加载器（保持不变）
# ========================
batch_size = 32
train_dataset = TensorDataset(trainX, trainY)
test_dataset = TensorDataset(testX, testY)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ========================
# 保持原有训练循环不变
# ========================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 1000
patience = 20
min_loss = np.inf
patience_counter = 0

train_losses = []
val_losses = []

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

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(test_loader.dataset)
    val_losses.append(val_loss)

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

model.load_state_dict(best_model)

# ========================
# 新增损失曲线绘制代码（放在训练循环结束后，预测代码前）
# ========================
plt.figure(figsize=(12, 6))

# 获取实际训练轮次（即列表长度）
actual_epochs = len(train_losses)

# 生成与实际轮次匹配的x轴数据
x_values = range(1, actual_epochs + 1)

# 绘制训练损失曲线（蓝色）
plt.plot(x_values, train_losses,
         color='blue', label='Train Loss')

# 绘制验证损失曲线（红色）
plt.plot(x_values, val_losses,
         color='red', linestyle='--', label='Validation Loss')

# 添加标题和标签
plt.title(f'Training and Validation Loss Curves ({actual_epochs} Epochs)', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)

# 添加网格线和图例
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# 调整x轴刻度（每5个epoch显示一个刻度）
step = max(1, actual_epochs // 10)  # 至少每10%显示一个刻度
plt.xticks(np.arange(1, actual_epochs + 1, step=step))

# 显示图像
plt.show()


# ========================
# 预测与评估（保持不变）
# ========================
# ========================
# 预测与评估（新增训练集预测）
# ========================
# 预测与评估（新增训练集预测）
model.eval()
with torch.no_grad():
    # 测试集预测
    testPredict = model(testX).numpy()
    testPredict = scaler_target.inverse_transform(testPredict).flatten()

    # 新增训练集预测
    trainPredict = model(trainX).numpy()
    trainPredict = scaler_target.inverse_transform(trainPredict).flatten()

# 反归一化真实值
trainY_actual = scaler_target.inverse_transform(trainY.numpy()).flatten()
testY_actual = scaler_target.inverse_transform(testY.numpy()).flatten()

# 创建训练集和测试集的 DataFrame
train_results = pd.DataFrame({
    'Train_Actual': trainY_actual,
    'Train_Predicted': trainPredict
})

test_results = pd.DataFrame({
    'Test_Actual': testY_actual,
    'Test_Predicted': testPredict
})

# 保存到Excel（使用不同的Sheet区分）
output_path = "D://vocation//xls//daochu//predictions.xlsx"
with pd.ExcelWriter(output_path) as writer:
    # 训练集结果
    train_results.to_excel(writer, sheet_name='Train_Results', index=False)
    # 测试集结果
    test_results.to_excel(writer, sheet_name='Test_Results', index=False)

print(f"预测结果已保存到 {output_path}")

# 计算 RMSE 指标
train_rmse = mean_squared_error(trainY_actual, trainPredict, squared=False)
test_rmse = mean_squared_error(testY_actual, testPredict, squared=False)

print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
# ========================
# 可视化对比（新增）
# ========================
plt.figure(figsize=(18, 6))

# 测试集对比
plt.subplot(1, 2, 1)
plt.plot(testY_actual, label='Actual', alpha=0.7)
plt.plot(testPredict, label='Predicted', alpha=0.7)
plt.title('Test Set Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Water Level (wl)')

# 训练集对比
plt.subplot(1, 2, 2)
plt.plot(trainY_actual, label='Actual', alpha=0.7)
plt.plot(trainPredict, label='Predicted', alpha=0.7)
plt.title('Train Set Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Water Level (wl)')

plt.tight_layout()
plt.show()