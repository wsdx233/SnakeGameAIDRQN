import matplotlib.pyplot as plt
import os

# 确保 plots 目录存在
if not os.path.exists('plots'):
    os.makedirs('plots')

def plot_scores(scores, mean_scores, save_path='plots/training_progress.png'):
    """
    绘制分数和平均分数，并将图表保存到指定路径。
    """
    # 创建一个新的图表实例，避免全局状态问题
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.set_title('Training Progress')
    ax.set_xlabel('Number of Games')
    ax.set_ylabel('Score')
    
    # 绘制数据
    ax.plot(scores, label='Score per Episode')
    ax.plot(mean_scores, label='Mean Score', linestyle='--')
    
    # 设置Y轴从0开始
    ax.set_ylim(ymin=0)
    
    # 添加图例
    ax.legend()
    
    # 在图上标记最新的分数和平均分
    if scores:
        ax.text(len(scores) - 1, scores[-1], str(scores[-1]))
    if mean_scores:
        ax.text(len(mean_scores) - 1, mean_scores[-1], f'{mean_scores[-1]:.2f}')
        
    # 保存图表到文件
    plt.savefig(save_path)
    
    # 关闭图表以释放内存
    plt.close(fig)