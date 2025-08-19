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
        
        # 用于混合精度训练
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP)

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
        
        # 从经验池中采样批次索引
        batch_indices = np.random.choice(
            np.arange(full_seq_len, len(self.memory)),
            size=config.BATCH_SIZE,
            replace=False
        )

        # 手动为批次构建序列，以避免numpy对象数组的类型问题
        s_seqs_list, a_seqs_list, r_seqs_list, ns_seqs_list, d_seqs_list = [], [], [], [], []

        for idx in batch_indices:
            # 提取一个完整的序列
            full_sequence = [self.memory[i] for i in range(idx - full_seq_len, idx)]
            
            # 解压元组序列
            s, a, r, ns, d = zip(*full_sequence)
            
            s_seqs_list.append(s)
            a_seqs_list.append(a)
            r_seqs_list.append(r)
            ns_seqs_list.append(ns)
            d_seqs_list.append(d)

        # 将列表转换为Numpy数组
        s_seqs = np.array(s_seqs_list, dtype=np.float32)
        a_seqs = np.array(a_seqs_list)
        r_seqs = np.array(r_seqs_list, dtype=np.float32)
        ns_seqs = np.array(ns_seqs_list, dtype=np.float32)
        d_seqs = np.array(d_seqs_list, dtype=bool)

        # 分割 burn-in 和 training 部分
        burn_in_state_batch = torch.from_numpy(s_seqs[:, :config.BURN_IN_LENGTH]).float().to(config.DEVICE)
        train_state_batch = torch.from_numpy(s_seqs[:, config.BURN_IN_LENGTH:]).float().to(config.DEVICE)
        burn_in_next_state_batch = torch.from_numpy(ns_seqs[:, :config.BURN_IN_LENGTH]).float().to(config.DEVICE)
        train_next_state_batch = torch.from_numpy(ns_seqs[:, config.BURN_IN_LENGTH:]).float().to(config.DEVICE)

        action_batch = torch.from_numpy(np.argmax(a_seqs[:, config.BURN_IN_LENGTH:], axis=2)).long().unsqueeze(2).to(config.DEVICE)
        reward_batch = torch.from_numpy(r_seqs[:, config.BURN_IN_LENGTH:]).float().unsqueeze(2).to(config.DEVICE)
        done_batch = torch.from_numpy(d_seqs[:, config.BURN_IN_LENGTH:]).float().unsqueeze(2).to(config.DEVICE)


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
        with torch.cuda.amp.autocast(enabled=config.USE_AMP):
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
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer) # 在梯度裁剪前unscale
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        