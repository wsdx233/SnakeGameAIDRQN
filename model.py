import torch
import torch.nn as nn
import config
import torch.nn.functional as F

class DRQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DRQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions

        # CNN for spatial feature extraction
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # 计算CNN输出的扁平化尺寸
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size - kernel_size + 2 * padding) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[1])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2])))
        linear_input_size = convw * convh * 64
        
        # LSTM for temporal feature processing
        self.lstm = nn.LSTM(input_size=linear_input_size, hidden_size=256, num_layers=1, batch_first=True)
        
        # Fully connected layers for Q-value output
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_actions)

    def forward(self, x, hidden):
        # x shape: (batch_size, sequence_length, C, H, W)
        batch_size, seq_len, C, H, W = x.size()
        
        # 将 batch 和 sequence 合并，以便通过CNN
        c_in = x.view(batch_size * seq_len, C, H, W)
        
        c_out = F.relu(self.conv1(c_in))
        c_out = F.relu(self.conv2(c_out))
        c_out = F.relu(self.conv3(c_out))
        
        # 恢复 batch 和 sequence 维度，为LSTM做准备
        lstm_in = c_out.view(batch_size, seq_len, -1)
        
        # LSTM处理
        lstm_out, hidden = self.lstm(lstm_in, hidden)
        
        # 只取序列最后的输出进行决策
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        # q_in shape: (batch_size, hidden_size)
        q_in = lstm_out[:, -1, :]

        # 全连接层输出Q值
        q_out = F.relu(self.fc1(q_in))
        q_values = self.fc2(q_out)
        
        return q_values, hidden

    def init_hidden(self, batch_size):
        # 初始化LSTM的隐藏状态和细胞状态
        # (num_layers, batch_size, hidden_size)
        return (torch.zeros(1, batch_size, self.lstm.hidden_size).to(config.DEVICE),
                torch.zeros(1, batch_size, self.lstm.hidden_size).to(config.DEVICE))
