
from pettingzoo.utils import ParallelEnv
from gymnasium.spaces import Discrete, Box
import numpy as np
import os
from scipy.ndimage import distance_transform_edt
import random
import json
from datetime import datetime

print("GridWorldEnv with enhanced monitoring and evaluation is loaded")

NUM_AGENTS = 2
GRID_SIZE = 40
NUM_ACTIONS = 9

# Action mapping: up and down, left and right, diagonal, stationary
ACTION_MAP = {
    0: (-1, 0),   
    1: (1, 0),    
    2: (0, -1),  
    3: (0, 1),    
    4: (-1, -1),  
    5: (-1, 1),   
    6: (1, -1),  
    7: (1, 1),    
    8: (0, 0),   
}

class EpisodeMonitor:
    "Single episode Monitor"
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.step_rewards = []
        self.positions_history = []
        self.actions_history = []
        self.distances_to_goal = []
        self.collision_events = []
        self.obstacle_penalties = []
        self.stuck_events = []
        self.success = False
        self.steps = 0
    
    def record_step(self, rewards, positions, actions, distances, infos):
        self.step_rewards.append(rewards.copy())
        self.positions_history.append(positions.copy())
        self.actions_history.append(actions.copy())
        self.distances_to_goal.append(distances.copy())
        self.steps += 1
        
        # Record Special events
        collision_count = 0
        obstacle_penalty = 0
        stuck_count = 0
        
        for agent, info in infos.items():
            if 'collision' in info:
                collision_count += 1
            if 'obstacle_penalty' in info:
                obstacle_penalty += info['obstacle_penalty']
            if 'stuck_penalty' in info:
                stuck_count += 1
        
        self.collision_events.append(collision_count)
        self.obstacle_penalties.append(obstacle_penalty)
        self.stuck_events.append(stuck_count)
    
    def finalize(self, success):
        self.success = success
        
    def get_summary(self):
        total_reward = sum([sum(step.values()) for step in self.step_rewards])
        mean_distance = np.mean([np.mean(list(dist.values())) for dist in self.distances_to_goal])
        total_collisions = sum(self.collision_events)
        total_obstacle_penalty = sum(self.obstacle_penalties)
        total_stuck_events = sum(self.stuck_events)
        
        return {
            'total_reward': total_reward,
            'mean_distance': mean_distance,
            'total_collisions': total_collisions,
            'total_obstacle_penalty': total_obstacle_penalty,
            'total_stuck_events': total_stuck_events,
            'steps': self.steps,
            'success': self.success,
            'success_rate': 1.0 if self.success else 0.0
        }

class TrainingMonitor:
    "Training Process Monitor"
    def __init__(self):
        self.episode_summaries = []
        self.best_success_rate = 0.0
        self.best_episode = -1
    
    def record_episode(self, episode_summary):
        self.episode_summaries.append(episode_summary)
        
        # Update Best performance
        if episode_summary['success']:
            current_success_rate = np.mean([s['success'] for s in self.episode_summaries[-100:]])
            if current_success_rate > self.best_success_rate:
                self.best_success_rate = current_success_rate
                self.best_episode = len(self.episode_summaries) - 1
    
    def get_statistics(self, window=100):
        if len(self.episode_summaries) == 0:
            return {}
        
        recent = self.episode_summaries[-window:]
        
        stats = {
            'mean_reward': np.mean([s['total_reward'] for s in recent]),
            'std_reward': np.std([s['total_reward'] for s in recent]),
            'mean_success_rate': np.mean([s['success_rate'] for s in recent]),
            'std_success_rate': np.std([s['success_rate'] for s in recent]),
            'mean_steps': np.mean([s['steps'] for s in recent]),
            'mean_collisions': np.mean([s['total_collisions'] for s in recent]),
            'mean_distance': np.mean([s['mean_distance'] for s in recent]),
            'episode_count': len(self.episode_summaries),
            'best_success_rate': self.best_success_rate,
            'best_episode': self.best_episode
        }
        
        return stats
    
    def save_to_file(self, filename):
        data = {
            'episode_summaries': self.episode_summaries,
            'timestamp': datetime.now().isoformat()
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        self.episode_summaries = data['episode_summaries']

def set_seed(seed):
    """Set random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

class GridWorldEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "GridWorld_v1"}

    def __init__(self, render_mode=None, seed=None, enable_monitoring=True):
        self.seed = seed
        if seed is not None:
            set_seed(seed)
            
        self.agents = [f"agent_{i}" for i in range(NUM_AGENTS)]
        self.possible_agents = self.agents[:]
        self.grid_size = GRID_SIZE
        self.render_mode = render_mode
        self.obs_radius = 3
        self.min_obstacle_dist_threshold = 2.0
        self.spawn_candidates = [
            np.array([16, 37]),
            np.array([31, 30]),
            np.array([24, 19]),
            np.array([20, 19]),
            np.array([19, 3]),
        ]
        
        
        self.enable_monitoring = enable_monitoring
        self.episode_monitor = EpisodeMonitor()
        self.training_monitor = TrainingMonitor()
        
        
        self.obstacles = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self._load_obstacles_from_file("obstacle_map.txt")
        self._update_distance_field()

        # Action & Observation space
        self.action_spaces = {agent: Discrete(NUM_ACTIONS) for agent in self.agents}
        obs_size = 6 + (2 * self.obs_radius + 1) ** 2
        self.observation_spaces = {
            agent: Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)
            for agent in self.agents
        }


        self.reach_reward = 20.0     # Increase the finish line rewards
        self.team_reward = 10.0      # Enhance cooperation incentives
        self.step_penalty = -0.02    # Increase the step count penalty
        self.collision_penalty = -2.0 


        self.reset()

    def _load_obstacles_from_file(self, filepath):
        if not os.path.exists(filepath):
            print(f"The obstacle map file {filepath} does not exist. By default, it is accessible.")
            return
        with open(filepath, "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.rstrip("\n")
            for j, ch in enumerate(line):
                if i < self.grid_size and j < self.grid_size:
                    if ch == '#':
                        self.obstacles[i, j] = True

    def reset(self, seed=None, options=None):
        if seed is not None:
            set_seed(seed)
            
        
        if self.enable_monitoring:
            self.episode_monitor.reset()
            
        self.agents = self.possible_agents[:]
        self.positions = {}
        self.terminated = {agent: False for agent in self.agents}
        self.steps = 0

        # Randomly select an endpoint
        while True:
            self.goal = self.spawn_candidates[np.random.choice(len(self.spawn_candidates))]
            if not self.obstacles[self.goal[1], self.goal[0]]:
                break

        # The starting point is randomly selected from the candidates as two points that do not equal the ending point
        possible_starts = [pos for pos in self.spawn_candidates if not np.array_equal(pos, self.goal)]
        selected_starts_idx = np.random.choice(len(possible_starts), size=NUM_AGENTS, replace=False)
        for i, agent in enumerate(self.agents):
            self.positions[agent] = possible_starts[selected_starts_idx[i]]

        self._update_distance_field()

        # The starting point is randomly selected from the candidates as two points that do not equal the ending point离
        self.prev_positions = {agent: np.copy(self.positions[agent]) for agent in self.agents}
        self.prev_dist_to_goal = {agent: np.linalg.norm(self.goal - self.positions[agent]) for agent in self.agents}

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        return observations, {agent: {} for agent in self.agents}

    def step(self, actions):
        rewards = {agent: 0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        self.steps += 1

        # update location
        for agent, action in actions.items():
            if not self.terminated[agent]:
                move = ACTION_MAP[action]
                new_pos = self.positions[agent] + np.array(move)
                new_pos = np.clip(new_pos, 0, self.grid_size - 1)
                if self.obstacles[new_pos[1], new_pos[0]]:
                    new_pos = self.positions[agent]
                self.positions[agent] = new_pos

        # Obstacle distance penalty
        for agent in self.agents:
            if not self.terminated[agent]:
                min_dist = self._get_min_obstacle_distance(self.positions[agent])
                if min_dist < self.min_obstacle_dist_threshold:
                    dist_penalty = -0.5 * (1.0 - min_dist / self.min_obstacle_dist_threshold)
                    rewards[agent] += dist_penalty
                    infos[agent]["obstacle_penalty"] = dist_penalty

        
        all_reached = True
        for i, agent in enumerate(self.agents):
            pos = self.positions[agent]
            if self.terminated[agent]:
                rewards[agent] = 0.0
            else:
                new_dist = np.linalg.norm(self.goal - pos)
                reached = new_dist < 1.5

                if reached:
                    rewards[agent] += self.reach_reward
                    self.terminated[agent] = True
                else:
                    # Distance difference reward (positive)
                    dist_diff = self.prev_dist_to_goal[agent] - new_dist
                    rewards[agent] += dist_diff * 2.0
                    rewards[agent] += self.step_penalty

                    # Stuck penalty
                    if np.array_equal(pos, self.prev_positions[agent]):
                        rewards[agent] -= 0.05
                        infos[agent]["stuck_penalty"] = True

                    # collision detection
                    for j, other_agent in enumerate(self.agents):
                        if i < j and np.array_equal(pos, self.positions[other_agent]):
                            rewards[agent] += self.collision_penalty
                            rewards[other_agent] += self.collision_penalty
                            infos[agent]["collision"] = True
                            infos[other_agent]["collision"] = True
                            break

                all_reached = all_reached and reached

        if all_reached:
            for agent in self.agents:
                rewards[agent] += self.team_reward
                infos[agent]["team_reward"] = True

        if self.steps >= 100:
            for agent in self.agents:
                truncations[agent] = True

        
        for agent in self.agents:
            self.prev_positions[agent] = np.copy(self.positions[agent])
            self.prev_dist_to_goal[agent] = np.linalg.norm(self.goal - self.positions[agent])

        # Record monitoring data
        if self.enable_monitoring:
            current_distances = {agent: np.linalg.norm(self.goal - self.positions[agent]) 
                               for agent in self.agents}
            self.episode_monitor.record_step(rewards, self.positions.copy(), 
                                           actions, current_distances, infos)

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        terminations = self.terminated.copy()
        return observations, rewards, terminations, truncations, infos

    def _update_distance_field(self):
        "Precomputed Obstacle Distance Field"
        self.distance_field = distance_transform_edt(~self.obstacles)

    def _get_min_obstacle_distance(self, pos):
        """Obtain the distance from the current position to the nearest obstacle"""
        x, y = int(pos[0]), int(pos[1])
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            return self.distance_field[y, x]
        return 0.0

    def _get_local_obstacles(self, pos):
        """Extract local obstacle observations"""
        local_obs = np.zeros((2*self.obs_radius+1, 2*self.obs_radius+1))
        for dy in range(-self.obs_radius, self.obs_radius+1):
            for dx in range(-self.obs_radius, self.obs_radius+1):
                x, y = pos[0] + dx, pos[1] + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    local_obs[dy+self.obs_radius, dx+self.obs_radius] = self.obstacles[y, x]
        return local_obs.flatten()
    
    def _get_observation(self, agent):
        """The improved observation vector"""
        self_pos = self.positions[agent]
        
        other_agent = next(a for a in self.agents if a != agent)
        obs = np.concatenate([
            self_pos / self.grid_size,
            (self.positions[other_agent] - self_pos) / self.grid_size,
            (self.goal - self_pos) / self.grid_size
        ])
        
        local_obs = self._get_local_obstacles(self_pos)
        
        return np.concatenate([obs, local_obs]).astype(np.float32)

    def render(self):
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.obstacles[y, x]:
                    grid[y, x] = "#"
        gx, gy = self.goal
        grid[gy, gx] = "G"
        for agent, (x, y) in self.positions.items():
            grid[y, x] = agent[-1]
        print("\n".join(" ".join(row) for row in grid))
        print(f"Steps: {self.steps}")

    def close(self):
        pass

    # 新增的监控相关方法
    def get_episode_summary(self):
        if not self.enable_monitoring:
            return {}
        
        success = all(self.terminated.values())
        self.episode_monitor.finalize(success)
        return self.episode_monitor.get_summary()
    
    def record_episode_to_training_monitor(self):
        if not self.enable_monitoring:
            return
        
        summary = self.get_episode_summary()
        self.training_monitor.record_episode(summary)
        return summary
    
    def get_training_statistics(self, window=100):
        return self.training_monitor.get_statistics(window)
    
    def save_training_data(self, filename):
  
        self.training_monitor.save_to_file(filename)
    
    def load_training_data(self, filename):
   
        self.training_monitor.load_from_file(filename)

def make_env(env_id=None, seed=None, enable_monitoring=True):
    return GridWorldEnv(seed=seed, enable_monitoring=enable_monitoring)