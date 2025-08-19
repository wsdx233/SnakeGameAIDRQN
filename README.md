# DRQN 贪吃蛇 (PyTorch)

这是一个使用深度循环Q网络 (Deep Recurrent Q-Network, DRQN) 和 PyTorch 实现的贪吃蛇AI项目。

DRQN 结构: `输入游戏画面 -> CNN 提取空间特征 -> LSTM 处理时序特征 -> 输出Q值`

## 项目结构

```
.
├── requirements.txt
└── src
    ├── agent.py         # Agent逻辑 (动作选择, 经验回放, 学习)
    ├── config.py        # 所有超参数和配置
    ├── model.py         # DRQN 网络模型定义
    ├── plot.py          # 绘图工具
    ├── snake_game.py    # 贪吃蛇游戏环境
    └── train.py         # 主训练脚本
```

## 如何运行

1.  **创建虚拟环境 (推荐)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

3.  **开始训练**
    ```bash
    python src/train.py
    ```

训练过程中，模型权重会保存在 `models/` 目录下，训练分数图会保存在 `plots/` 目录下。游戏的可视化窗口将显示AI的实时操作。
