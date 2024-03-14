import torch

# 训练参数
actor_lr = 1e-4
critic_lr = 1e-4
gamma = 0.9
lmbda = 0.95
epochs = 10
eps = 0.3
device = torch.device("cuda")

# 输入通道数,这里我把棋盘上每个点都编码为12维
in_channels = 12
out_channels = 12  # 这个参数是没用的

# CNN层的输出特征数 - 这也是1x1卷积层的输出通道数
out_features_cnn = 12

# MultiHeadAttention的相关参数
num_hidden = 64

# 注意力头数
num_heads = 4

# Dropout率
dropout = 0.1

# Base最终输出特征数
final_out_features = 32
