from pettingzoo.utils import ParallelEnv
from gymnasium.spaces import Discrete, Box
import numpy as np

print("TrainEnv.py is loaded")

NUM_AGENTS = 3
GRID_SIZE = 5
NUM_ACTIONS = 9

# 9个动作映射：上、下、左、右、↖↗↘↙、不动
ACTION_MAP = {
    0: (-1, 0),   # 上
    1: (1, 0),    # 下
    2: (0, -1),   # 左
    3: (0, 1),    # 右
    4: (-1, -1),  # ↖
    5: (-1, 1),   # ↗
    6: (1, -1),   # ↙
    7: (1, 1),    # ↘
    8: (0, 0),    # 原地
}

class GridWorldEnv(ParallelEnv):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        self.agents = [f"agent_{i}" for i in range(NUM_AGENTS)]
        self.possible_agents = self.agents[:]

        # 动作空间
        self.action_spaces = {agent: Discrete(NUM_ACTIONS) for agent in self.agents}

        # 观察空间，每个智能体的观察空间是一个二维空间，范围为 [0, GRID_SIZE-1]
        self.observation_spaces = {
            agent: Box(low=0, high=GRID_SIZE - 1, shape=(2,), dtype=np.int32)
            for agent in self.agents
        }

        self.grid_size = GRID_SIZE
        self.render_mode = render_mode
        self.reset()

    def reset(self, seed=None, options=None):
        """环境重置方法，初始化智能体的位置"""
        self.agents = self.possible_agents[:]
        self.positions = {
            agent: np.random.randint(0, self.grid_size, size=(2,))
            for agent in self.agents
        }
        observations = {agent: self.positions[agent] for agent in self.agents}
        self.terminated = {agent: False for agent in self.agents}
        return observations

    def step(self, actions):
        """执行一步操作，更新状态和奖励"""
        rewards = {}
        observations = {}
        terminations = {}
        truncations = {}
        infos = {}

        # 更新每个智能体的位置
        for agent, action in actions.items():
            move = ACTION_MAP[action]
            pos = self.positions[agent] + np.array(move)
            pos = np.clip(pos, 0, self.grid_size - 1)  # 确保位置在网格范围内
            self.positions[agent] = pos

        # 计算每个智能体的奖励
        for agent in self.agents:
            # 奖励函数：靠近(0,0) 给正奖励
            distance = np.linalg.norm(self.positions[agent])  # 计算与原点的距离
            rewards[agent] = -distance  # 奖励：距离越远奖励越小
            observations[agent] = self.positions[agent]
            terminations[agent] = False  # 默认环境没有结束
            truncations[agent] = False   # 没有截断
            infos[agent] = {}  # 可以放置更多的元数据

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """渲染当前的网格状态"""
        grid = np.full((self.grid_size, self.grid_size), ".")  # 创建空网格
        for agent, pos in self.positions.items():
            x, y = pos
            grid[x, y] = agent[-1]  # 用智能体编号表示位置
        print("\n".join(" ".join(row) for row in grid))
        print()
