import pygame
import random
from collections import deque
import numpy as np
import config

# 定义颜色
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

BLOCK_SIZE = 20

class SnakeGame:
    def __init__(self, w=config.GRID_SIZE, h=config.GRID_SIZE):
        self.w = w
        self.h = h
        # Pygame 初始化 (可选，用于可视化)
        self.display = None
        self.clock = None

    def _init_pygame(self):
        pygame.init()
        self.display = pygame.display.set_mode((self.w * BLOCK_SIZE, self.h * BLOCK_SIZE))
        pygame.display.set_caption('DRQN Snake')
        self.clock = pygame.time.Clock()

    def reset(self):
        self.head = (self.w // 2, self.h // 2)
        
        # 根据随机方向正确初始化蛇身
        self.direction = random.choice(['U', 'D', 'L', 'R'])
        if self.direction == 'U':
            self.snake = deque([self.head, (self.head[0], self.head[1]+1), (self.head[0], self.head[1]+2)])
        elif self.direction == 'D':
            self.snake = deque([self.head, (self.head[0], self.head[1]-1), (self.head[0], self.head[1]-2)])
        elif self.direction == 'L':
            self.snake = deque([self.head, (self.head[0]+1, self.head[1]), (self.head[0]+2, self.head[1])])
        elif self.direction == 'R':
            self.snake = deque([self.head, (self.head[0]-1, self.head[1]), (self.head[0]-2, self.head[1])])
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        return self.get_state()

    def _place_food(self):
        while True:
            x = random.randint(0, self.w - 1)
            y = random.randint(0, self.h - 1)
            self.food = (x, y)
            if self.food not in self.snake:
                break

    def get_state(self):
        # 将游戏状态转换为一个 GRID_SIZE x GRID_SIZE 的 numpy 数组
        # 0: 空白, 1: 蛇身, 2: 蛇头, 3: 食物
        state = np.zeros((self.w, self.h), dtype=np.float32)
        if self.food:
            state[self.food[0], self.food[1]] = 3
        for part in self.snake:
            state[part[0], part[1]] = 1
        state[self.head[0], self.head[1]] = 2
        return state.T # 转置以匹配 (H, W) 习惯

    def step(self, action):
        # action: [1, 0, 0] -> 直行, [0, 1, 0] -> 右转, [0, 0, 1] -> 左转
        self.frame_iteration += 1

        # 计算移动前与食物的距离
        dist_before = np.linalg.norm(np.array(self.head) - np.array(self.food))

        # 1. 确定新方向
        # 顺序: U, R, D, L
        dirs = ['U', 'R', 'D', 'L']
        current_dir_idx = dirs.index(self.direction)

        if np.array_equal(action, [1, 0, 0]): # 直行
            new_dir = self.direction
        elif np.array_equal(action, [0, 1, 0]): # 右转
            new_dir_idx = (current_dir_idx + 1) % 4
            new_dir = dirs[new_dir_idx]
        else:  # [0, 0, 1] 左转
            new_dir_idx = (current_dir_idx - 1) % 4
            new_dir = dirs[new_dir_idx]
        self.direction = new_dir

        # 2. 计算新蛇头位置
        x, y = self.head
        if self.direction == 'U': y -= 1
        elif self.direction == 'D': y += 1
        elif self.direction == 'L': x -= 1
        elif self.direction == 'R': x += 1
        self.head = (x, y)

        # 3. 检查游戏是否结束
        game_over = False
        # 简化奖励：移除基于蛇身长度的存活惩罚和靠近/远离食物的奖励
        if self._is_collision():
            game_over = True
            reward = -10
            # 当游戏结束时，我们不能调用 get_state()，因为蛇头可能已经越界。
            # 返回一个与 get_state() 输出形状相同的零数组作为最终状态。
            state = np.zeros((self.w, self.h), dtype=np.float32)
            return state.T, reward, game_over, self.score

        # 4. 检查是否吃到食物并调整奖励
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            reward = -0.01 # 每走一步给予微小的惩罚，鼓励效率
            self.snake.pop()

        self.snake.appendleft(self.head)
        
        return self.get_state(), reward, game_over, self.score

    def _is_collision(self, pt=None):
        if pt is None: pt = self.head
        # 撞墙
        if pt[0] >= self.w or pt[0] < 0 or pt[1] >= self.h or pt[1] < 0:
            return True
        # 撞自己
        if pt in list(self.snake)[1:]:
            return True
        return False

    def render(self):
        if self.display is None:
            self._init_pygame()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt[0]*BLOCK_SIZE, pt[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt[0]*BLOCK_SIZE+4, pt[1]*BLOCK_SIZE+4, 12, 12))
        
        # 蛇头用绿色表示
        pygame.draw.rect(self.display, GREEN, pygame.Rect(self.head[0]*BLOCK_SIZE, self.head[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food[0]*BLOCK_SIZE, self.food[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        text = pygame.font.Font(None, 25).render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        self.clock.tick(config.FPS)
