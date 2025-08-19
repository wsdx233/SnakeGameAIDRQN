import torch

# --- 设备配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 游戏配置 ---
GRID_SIZE = 10  # 游戏区域大小 (GRID_SIZE x GRID_SIZE)
FPS = 20 # 游戏可视化时的帧率
RENDER = True # 是否在训练时渲染游戏画面

# --- 模型超参数 ---
# DRQN 输入为 (N, S, C, H, W) -> (批大小, 序列长度, 通道数, 高, 宽)
SEQUENCE_LENGTH = 4  # 输入到LSTM的序列长度
BURN_IN_LENGTH = 10   # “预热”序列的长度，用于生成隐藏状态

# --- 训练超参数 ---
NUM_EPISODES = 50000        # 总训练回合数
MEMORY_SIZE = 50000         # 经验回放池大小
BATCH_SIZE = 128            # 每批训练的样本数
GAMMA = 0.99                # 折扣因子
LEARNING_RATE = 0.0005      # 学习率

# --- Epsilon-Greedy 策略参数 ---
EPSILON_START = 1.0         # Epsilon 初始值
EPSILON_END = 0.01          # Epsilon 最终值
EPSILON_DECAY = 20000       # Epsilon 衰减步数 (调整后)

# --- 更新与保存 ---
TARGET_UPDATE_FREQ = 100    # 目标网络更新频率 (每N个episodes)
MODEL_SAVE_FREQ = 500       # 模型保存频率 (每N个episodes)
