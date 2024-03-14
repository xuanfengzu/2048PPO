import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *


class SimpleCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=4, out_channels=in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        # 通过第一个3x3卷积层
        x = self.sigmoid(self.conv1(x))
        # 通过第二个1x1卷积层
        x = self.relu(self.conv2(x))

        # 展平操作
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)

        return x


class DotProdycrAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self._attention_weights = F.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self._attention_weights), values)


class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hidden, num_heads, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProdycrAttention(dropout)
        self.W_k = nn.Linear(key_size, num_hidden, bias=bias)
        self.W_q = nn.Linear(query_size, num_hidden, bias=bias)
        self.W_v = nn.Linear(value_size, num_hidden, bias=bias)
        self.W_o = nn.Linear(num_hidden, num_hidden, bias=bias)

    def forward(self, queries, keys, values):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        output = self.attention(queries, keys, values)

        output_concat = transpose_output(output, self.num_heads)

        return self.W_o(output_concat)


class BaseNet(nn.Module):
    def __init__(self, in_channels, num_hidden, num_heads, out_features, dropout, out_channels, bias=False):
        super(BaseNet, self).__init__()
        self.simple_cnn = SimpleCNN(in_channels, out_channels)

        qkv_size = in_channels
        self.multi_head_attention = MultiHeadAttention(
            key_size=qkv_size, query_size=qkv_size, value_size=qkv_size,
            num_hidden=num_hidden, num_heads=num_heads, dropout=dropout, bias=bias
        )

        self.norm = nn.LayerNorm(num_hidden)

        self.linear_1 = nn.Linear(num_hidden, 128)
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(128, 64)
        self.relu_2 = nn.ReLU()
        self.linear_3 = nn.Linear(64, out_features)
        self.relu_3 = nn.ReLU()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        cnn_output_1 = self.simple_cnn(x)

        cnn_output = cnn_output_1 + \
            x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)

        attention_output = self.multi_head_attention(
            cnn_output, cnn_output, cnn_output)

        norm_output = self.norm(attention_output)

        pooled_output = norm_output.mean(dim=1)

        x_1 = self.relu_1(self.linear_1(pooled_output))
        x_2 = self.relu_2(self.linear_2(x_1))
        final_output = self.relu_3(self.linear_3(x_2))

        return final_output


class PolicyNet(nn.Module):
    def __init__(self, in_channels, num_hidden, num_heads, out_features, dropout, out_channels, bias=False, action_dim=4):
        super().__init__()
        self.basenet = BaseNet(
            in_channels, num_hidden, num_heads, out_features, dropout, out_channels, bias)
        self.fc = nn.Linear(out_features, action_dim)

    def forward(self, x):
        x = F.relu(self.basenet(x))
        return F.softmax(self.fc(x), dim=-1)


class ValueNet(nn.Module):
    def __init__(self, in_channels, num_hidden, num_heads, out_features, dropout, out_channels, bias=False):
        super().__init__()
        self.basenet = BaseNet(
            in_channels, num_hidden, num_heads, out_features, dropout, out_channels, bias)
        self.fc = nn.Linear(out_features, 1)

    def forward(self, x):
        x = F.relu(self.basenet(x))
        return self.fc(x)
