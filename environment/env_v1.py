from pettingzoo.utils import ParallelEnv
from gymnasium.spaces import Discrete, Box
import numpy as np

print("GridWorldEnv with shared goal is loaded")

NUM_AGENTS = 2
GRID_SIZE = 30
NUM_ACTIONS = 9

# 9 个动作映射：上下左右、对角、不动
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

        self.grid_size = GRID_SIZE
        self.render_mode = render_mode

        self.action_spaces = {agent: Discrete(NUM_ACTIONS) for agent in self.agents}
        # 观测空间：自身位置 (2), 另一个agent相对位置 (2), 公共目标相对位置 (2)
        self.observation_spaces = {
            agent: Box(low=-GRID_SIZE, high=GRID_SIZE, shape=(6,), dtype=np.float32)
            for agent in self.agents
        }

        self.reset()

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.positions = {}
        self.terminated = {agent: False for agent in self.agents}

        # 创建一个公共目标点，确保不与任意起始点重复
        while True:
            common_goal = np.random.randint(0, self.grid_size, size=(2,))
            valid = True
            temp_positions = []
            for _ in range(NUM_AGENTS):
                while True:
                    pos = np.random.randint(0, self.grid_size, size=(2,))
                    if not np.array_equal(pos, common_goal):
                        break
                temp_positions.append(pos)
            if all(not np.array_equal(p, common_goal) for p in temp_positions):
                break

        self.goal = common_goal
        for i, agent in enumerate(self.agents):
            self.positions[agent] = temp_positions[i]

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        return observations

    def step(self, actions):
        rewards = {}
        observations = {}
        terminations = {}
        truncations = {}
        infos = {}

        # 更新每个 agent 的位置（未完成的才移动）
        for agent, action in actions.items():
            if self.terminated[agent]:
                continue  # 已完成，不移动
            move = ACTION_MAP[action]
            new_pos = self.positions[agent] + np.array(move)
            new_pos = np.clip(new_pos, 0, self.grid_size - 1)
            self.positions[agent] = new_pos

        # 计算观测和奖励
        for agent in self.agents:
            pos = self.positions[agent]
            goal = self.goal  # 所有 agent 共用一个目标点

            if self.terminated[agent]:
                # 已完成，位置不变，奖励为 0
                reward = 0.0
                done = True
            else:
                dist = np.linalg.norm(goal - pos)
                reward = -dist  # 奖励为负距离
                done = np.array_equal(pos, goal)
                self.terminated[agent] = done

            observations[agent] = self._get_observation(agent)
            rewards[agent] = reward
            terminations[agent] = done
            truncations[agent] = False
            infos[agent] = {"goal": goal, "position": pos}

        return observations, rewards, terminations, truncations, infos


    def _get_observation(self, agent):
        self_pos = self.positions[agent]
        other_agent = [a for a in self.agents if a != agent][0]
        other_pos = self.positions[other_agent]
        relative_to_other = other_pos - self_pos
        relative_to_goal = self.goal - self_pos
        return np.concatenate([self_pos, relative_to_other, relative_to_goal]).astype(np.float32)

    def render(self):
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)

        gx, gy = self.goal
        grid[gx, gy] = "G"  # 目标点

        for agent, pos in self.positions.items():
            x, y = pos
            grid[x, y] = agent[-1]  # 用编号表示 agent_0 => 0

        print("\n".join(" ".join(row) for row in grid))
        print()

    def close(self):
        pass


# ✅ PettingZoo 环境工厂函数
def make_env(env_id=None):
    return GridWorldEnv()
