import torch
import numpy as np
from collections import deque
import os
from tqdm import tqdm

import config
from snake_game import SnakeGame
from agent import Agent
from plot import plot_scores

def train_no_display():
    # --- 无显示模式配置 ---
    config.RENDER = False
    print("Running in no-display mode. Pygame rendering is disabled.")

    # 创建目录
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # 初始化
    game = SnakeGame(w=config.GRID_SIZE, h=config.GRID_SIZE)
    # 输入 shape: (C, H, W)
    input_shape = (1, config.GRID_SIZE, config.GRID_SIZE)
    num_actions = 3 # [直行, 右转, 左转]
    agent = Agent(input_shape, num_actions)
    
    scores = []
    mean_scores = []
    total_score = 0
    
    # 状态序列，用于DRQN输入
    state_sequence = deque(maxlen=config.SEQUENCE_LENGTH)

    print(f"开始训练，设备: {config.DEVICE}")

    try:
        for episode in tqdm(range(1, config.NUM_EPISODES + 1)):
            # 重置环境和序列
            state = game.reset()
            state_sequence.clear()
            for _ in range(config.SEQUENCE_LENGTH):
                # 用初始状态填充序列
                state_sequence.append(state.reshape(input_shape))
            
            agent.reset_hidden_state() # 每个回合开始时重置LSTM隐藏状态
            
            done = False
            steps_in_episode = 0
            while not done:
                # 1. 渲染游戏 (在无显示模式下禁用)
                # if config.RENDER:
                #     game.render()

                # 2. Agent选择动作
                current_sequence_np = np.array(state_sequence)
                action = agent.select_action(current_sequence_np)

                # 3. 在环境中执行动作
                next_state, reward, done, score = game.step(action)
                
                # 验证修复：确保游戏不会在第一步就结束
                if steps_in_episode == 0:
                    assert not done, "游戏在第一步就结束了！请检查 snake_game.py 的 reset 和 step 逻辑。"

                # 4. 存储经验
                next_state_reshaped = next_state.reshape(input_shape)
                # 存储经验时，我们存储的是单个状态，而不是序列
                # ReplayBuffer会负责从这些单步经验中构建序列
                agent.store_transition(state.reshape(input_shape), action, reward, next_state_reshaped, done)
                
                # 更新当前状态
                state = next_state
                
                # 5. 更新状态序列
                state_sequence.append(next_state_reshaped)

                # 6. Agent学习
                agent.learn()
                
                steps_in_episode += 1

                if done:
                    scores.append(score)
                    total_score += score
                    mean_score = total_score / episode
                    mean_scores.append(mean_score)
                    
                    tqdm.write(f'Episode {episode}, Score: {score}, Mean Score: {mean_score:.2f}, Steps: {steps_in_episode}')
                    

            # 更新目标网络
            if episode % config.TARGET_UPDATE_FREQ == 0:
                agent.update_target_net()
                tqdm.write("--- Target Network Updated ---")

            # 保存模型
            if episode % config.MODEL_SAVE_FREQ == 0:
                torch.save(agent.policy_net.state_dict(), f'models/drqn_snake_episode_{episode}.pth')
                tqdm.write(f"--- Model Saved at Episode {episode} ---")
    
    finally:
        # 确保在退出时保存最终模型和图表
        print("\n训练结束或中断。正在保存最终模型和图表...")
        final_model_path = f'models/drqn_snake_final.pth'
        torch.save(agent.policy_net.state_dict(), final_model_path)
        print(f"最终模型已保存至 {final_model_path}")
        
        plot_scores(scores, mean_scores, save_path='plots/training_progress_final.png')
        print("最终训练图表已保存。")

if __name__ == '__main__':
    train_no_display()