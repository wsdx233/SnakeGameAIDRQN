import torch
import numpy as np
from collections import deque
import os
from tqdm import tqdm

import config
from snake_game import SnakeGame
from agent import Agent
from plot import plot_scores

def train():
    # 检查点文件路径
    CHECKPOINT_PATH = 'models/training_checkpoint.pth'

    # 创建目录
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # 初始化
    game = SnakeGame(w=config.GRID_SIZE, h=config.GRID_SIZE)
    # 输入 shape: (C, H, W)
    input_shape = (1, config.GRID_SIZE, config.GRID_SIZE)
    num_actions = 3  # [直行, 右转, 左转]
    agent = Agent(input_shape, num_actions)

    scores = []
    mean_scores = []
    total_score = 0
    start_episode = 1

    # 尝试加载检查点
    if os.path.exists(CHECKPOINT_PATH):
        try:
            print(f"正在从 {CHECKPOINT_PATH} 加载检查点...")
            checkpoint = torch.load(CHECKPOINT_PATH)
            agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            agent.memory = checkpoint['memory']
            agent.steps_done = checkpoint['steps_done']
            start_episode = checkpoint['episode'] + 1
            scores = checkpoint['scores']
            mean_scores = checkpoint['mean_scores']
            total_score = checkpoint['total_score']
            print(f"检查点加载完毕。将从 Episode {start_episode} 继续训练。")
        except (KeyError, EOFError) as e:
            print(f"加载检查点失败 ({e})，文件可能已损坏。将从头开始训练。")
            # 如果检查点文件损坏或不完整，则删除它
            os.remove(CHECKPOINT_PATH)

    # 状态序列，用于DRQN输入
    state_sequence = deque(maxlen=config.SEQUENCE_LENGTH)

    print(f"开始训练，设备: {config.DEVICE}")

    try:
        for episode in tqdm(range(start_episode, config.NUM_EPISODES + 1)):
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
                # 1. 渲染游戏 (可选)
                if config.RENDER:
                    game.render()

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

            # 保存检查点
            if episode % config.MODEL_SAVE_FREQ == 0:
                checkpoint = {
                    'episode': episode,
                    'policy_net_state_dict': agent.policy_net.state_dict(),
                    'target_net_state_dict': agent.target_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'memory': agent.memory,
                    'steps_done': agent.steps_done,
                    'scores': scores,
                    'mean_scores': mean_scores,
                    'total_score': total_score,
                }
                torch.save(checkpoint, CHECKPOINT_PATH)
                # 为了兼容旧的逻辑，仍然可以单独保存模型
                torch.save(agent.policy_net.state_dict(), f'models/drqn_snake_episode_{episode}.pth')
                tqdm.write(f"--- Checkpoint and Model Saved at Episode {episode} ---")
    
    finally:
        # 确保在退出时保存最终模型和图表
        print("\n训练结束或中断。正在保存最终模型和检查点...")
        final_model_path = f'models/drqn_snake_final.pth'
        torch.save(agent.policy_net.state_dict(), final_model_path)
        print(f"最终模型已保存至 {final_model_path}")

        # 保存最终检查点
        # 使用 len(scores) 来确定实际已经完成的 episode 数量
        # 如果是从头开始，len(scores) 就是 episode 数
        # 如果是续训，需要加上 start_episode
        # 但由于 scores 列表已经从检查点加载，它本身就包含了历史分数，所以直接用它的长度即可
        # total_score 也已经恢复，所以 mean_score 的计算也是连续的
        # episode 在循环中断时是最后一个完成的 episode + 1，所以保存时用 episode - 1
        # 但是如果循环一次都没跑，episode 就是 start_episode，scores 是空的
        # 一个更稳健的方法是基于 scores 的长度来计算
        if episode > start_episode: # 确保至少完成了一个 episode
             current_episode = episode -1
        else:
             current_episode = start_episode -1

        final_checkpoint = {
            'episode': current_episode,
            'policy_net_state_dict': agent.policy_net.state_dict(),
            'target_net_state_dict': agent.target_net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'memory': agent.memory,
            'steps_done': agent.steps_done,
            'scores': scores,
            'mean_scores': mean_scores,
            'total_score': total_score,
        }
        torch.save(final_checkpoint, CHECKPOINT_PATH)
        print(f"最终检查点已保存至 {CHECKPOINT_PATH}")
        
        plot_scores(scores, mean_scores, save_path='plots/training_progress_final.png')
        print("最终训练图表已保存。")

if __name__ == '__main__':
    train()
