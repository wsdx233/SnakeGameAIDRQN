import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
import config
from model import DRQN

class Agent:
    def __init__(self, input_shape, num_actions):
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.policy_net = DRQN(input_shape, num_actions).to(config.DEVICE)
        self.target_net = DRQN(input_shape, num_actions).to(config.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only for inference

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        
        # 经验回放池，存储 (state, action, reward, next_state, done)
        self.memory = deque(maxlen=config.MEMORY_SIZE)
        
        self.steps_done = 0
        self.hidden_state = None # 用于游戏中的隐藏状态传递

    def reset_hidden_state(self):
        self.hidden_state = self.policy_net.init_hidden(1)

    def select_action(self, state_sequence):
        # state_sequence: (seq_len, C, H, W)
        sample = random.random()
        eps_threshold = config.EPSILON_END + (config.EPSILON_START - config.EPSILON_END) * \
                        np.exp(-1. * self.steps_done / config.EPSILON_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # state_sequence needs to be (1, seq_len, C, H, W)
                state_tensor = torch.FloatTensor(state_sequence).unsqueeze(0).to(config.DEVICE)
                q_values, new_hidden = self.policy_net(state_tensor, self.hidden_state)
                self.hidden_state = new_hidden
                # 从序列Q值中选择最后一个时间步的Q值进行决策
                # q_values shape: (1, seq_len, num_actions) -> last_q_values shape: (1, num_actions)
                last_q_values = q_values[:, -1, :]
                action_idx = last_q_values.max(1)[1].item()
        else:
            # 探索时，也需要前向传播来更新隐藏状态
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_sequence).unsqueeze(0).to(config.DEVICE)
                _, new_hidden = self.policy_net(state_tensor, self.hidden_state)
                self.hidden_state = new_hidden
            action_idx = random.randrange(self.num_actions)

        action = np.zeros(self.num_actions)
        action[action_idx] = 1
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def can_sample(self):
        # 确保回放池中有足够的数据用于预热和训练
        return len(self.memory) >= config.BURN_IN_LENGTH + config.SEQUENCE_LENGTH + config.BATCH_SIZE

    def learn(self):
        if not self.can_sample():
            return

        # 1. 从经验池中采样一个批次的序列（包含burn-in和training部分）
        full_seq_len = config.BURN_IN_LENGTH + config.SEQUENCE_LENGTH
        batch_indices = random.sample(range(full_seq_len, len(self.memory)), config.BATCH_SIZE)

        burn_in_state_seqs, train_state_seqs = [], []
        burn_in_next_state_seqs, train_next_state_seqs = [], []
        action_seqs, reward_seqs, done_flags = [], [], []

        for i in batch_indices:
            full_sequence = list(self.memory)[i - full_seq_len : i]
            s, a, r, ns, d = zip(*full_sequence)

            # 分割序列
            burn_in_s = np.array(s[:config.BURN_IN_LENGTH])
            train_s = np.array(s[config.BURN_IN_LENGTH:])
            burn_in_ns = np.array(ns[:config.BURN_IN_LENGTH])
            train_ns = np.array(ns[config.BURN_IN_LENGTH:])

            burn_in_state_seqs.append(burn_in_s)
            train_state_seqs.append(train_s)
            burn_in_next_state_seqs.append(burn_in_ns)
            train_next_state_seqs.append(train_ns)
            
            # 动作、奖励、done取整个训练序列
            action_seqs.append(a[config.BURN_IN_LENGTH:])
            reward_seqs.append(r[config.BURN_IN_LENGTH:])
            done_flags.append(d[config.BURN_IN_LENGTH:])

        # 转换成Tensor
        burn_in_state_batch = torch.FloatTensor(np.array(burn_in_state_seqs)).to(config.DEVICE)
        train_state_batch = torch.FloatTensor(np.array(train_state_seqs)).to(config.DEVICE)
        burn_in_next_state_batch = torch.FloatTensor(np.array(burn_in_next_state_seqs)).to(config.DEVICE)
        train_next_state_batch = torch.FloatTensor(np.array(train_next_state_seqs)).to(config.DEVICE)

        # action_seqs is (B, SEQ_LEN, NUM_ACTIONS) -> (B, SEQ_LEN) -> (B, SEQ_LEN, 1)
        action_batch = torch.LongTensor(np.argmax(action_seqs, axis=2)).unsqueeze(2).to(config.DEVICE)
        # reward_seqs is (B, SEQ_LEN) -> (B, SEQ_LEN, 1)
        reward_batch = torch.FloatTensor(reward_seqs).unsqueeze(2).to(config.DEVICE)
        # done_flags is (B, SEQ_LEN) -> (B, SEQ_LEN, 1)
        done_batch = torch.FloatTensor(done_flags).unsqueeze(2).to(config.DEVICE)


        # 2. Burn-in: 生成隐藏状态
        with torch.no_grad():
            # 为 policy_net 生成隐藏状态 h0
            h0_init = self.policy_net.init_hidden(config.BATCH_SIZE)
            _, h0 = self.policy_net(burn_in_state_batch, h0_init)
            
            # 为 target_net 生成隐藏状态 h1
            h1_init = self.target_net.init_hidden(config.BATCH_SIZE)
            _, h1 = self.target_net(burn_in_next_state_batch, h1_init)

        # 3. 计算Q(s, a)
        # q_values: (B, SEQ_LEN, NUM_ACTIONS)
        q_values, _ = self.policy_net(train_state_batch, h0)
        # q_s_a: (B, SEQ_LEN, 1)
        q_s_a = q_values.gather(2, action_batch)

        # 4. 计算 V(s')
        with torch.no_grad():
            # q_next_values: (B, SEQ_LEN, NUM_ACTIONS)
            q_next_values, _ = self.target_net(train_next_state_batch, h1)
            # q_next_max: (B, SEQ_LEN, 1)
            q_next_max = q_next_values.max(2)[0].unsqueeze(2)
            
        # 5. 计算期望的Q值
        expected_q_s_a = reward_batch + (config.GAMMA * q_next_max * (1 - done_batch))

        # 6. 计算损失
        loss = F.smooth_l1_loss(q_s_a, expected_q_s_a)

        # 7. 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        