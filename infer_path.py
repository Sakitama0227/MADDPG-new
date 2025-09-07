import os
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from environment.env_v3 import make_env
from maddpg.MADDPG import MADDPG
from localization import get_current_position, start_localization_node

from scipy.ndimage import gaussian_filter1d

GRID_SIZE = 40
POSITION_TIMEOUT = 10.0
POSITION_RETRY_INTERVAL = 0.5
OBSTACLE_FILE = "obstacle_map.txt"


def load_obstacles(file_path):
    obstacles = set()
    if not os.path.exists(file_path):
        print(f"[WARN] Obstacle file not found: {file_path}")
        return obstacles
    with open(file_path, 'r') as f:
        for y, line in enumerate(f.readlines()):
            for x, char in enumerate(line.strip("\n")):
                if char == "#":
                    obstacles.add((x, y))
    print(f"[INFO] Loaded {len(obstacles)} obstacles")
    return obstacles


def smooth_and_resample(path, step=0.5, sigma=1.0, obstacles=None):
    path = np.array(path, dtype=float)
    if path.ndim == 1:
        path = path.reshape(1, -1)
    if len(path) < 3:
        return path.tolist()
    
    # Gaussian smoothing
    xs = gaussian_filter1d(path[:, 0], sigma=sigma)
    ys = gaussian_filter1d(path[:, 1], sigma=sigma)
    smooth_path = np.vstack((xs, ys)).T
    
    # Resampling
    distances = np.cumsum(np.r_[0, np.linalg.norm(np.diff(smooth_path, axis=0), axis=1)])
    new_distances = np.arange(0, distances[-1], step)
    new_x = np.interp(new_distances, distances, smooth_path[:, 0])
    new_y = np.interp(new_distances, distances, smooth_path[:, 1])
    resampled_path = np.vstack((new_x, new_y)).T

    # Obstacle check
    if obstacles:
        for gx, gy in np.round(resampled_path).astype(int):
            if (gx, gy) in obstacles:
                print(f"[WARN] Smoothed path passes through obstacle ({gx}, {gy}), using original path")
                return path.tolist()
    
    return resampled_path.tolist()


def visualize_paths(paths, obstacles):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, GRID_SIZE-0.5)
    ax.set_ylim(GRID_SIZE-0.5, -0.5)  # origin at top-left
    ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1))
    ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1))
    ax.grid(True, color='lightgray', linewidth=0.5)

    # Draw obstacles
    for (x, y) in obstacles:
        ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color='black'))

    colors = ['red', 'blue', 'green', 'orange']
    for i, (agent, path) in enumerate(paths.items()):
        path = np.array(path, dtype=float)

        # Ensure 2D array (N, 2)
        if path.ndim == 1:
            path = path.reshape(1, -1)
        if path.shape[1] != 2:
            print(f"[WARN] {agent} path format error: {path.shape}, skipping")
            continue

        ax.plot(path[:, 0], path[:, 1], marker='o', markersize=2,
                color=colors[i % len(colors)], label=agent)
        ax.scatter(path[0, 0], path[0, 1], color='yellow', edgecolors='black', s=80, label=f"{agent} start")
        ax.scatter(path[-1, 0], path[-1, 1], color='lime', edgecolors='black', s=80, label=f"{agent} goal")

    ax.legend()
    plt.title("Inferred Paths")
    plt.show()


# Initialize ROS2 at the start of infer_path
def infer_path(model_path, goal_positions=None):
    # Ensure ROS2 is initialized only once
    start_localization_node()

    env = make_env()
    dim_info = {agent: (env.observation_spaces[agent].shape[0], env.action_spaces[agent].n) 
               for agent in env.agents}
    print("[INFO] Loading trained model...")
    maddpg = MADDPG.load(dim_info, model_path)

    # Load obstacles
    obstacles = load_obstacles(OBSTACLE_FILE)

    # Get agent positions
    agent_info = {}
    start_time = time.time()
    active_agents = []
    while time.time() - start_time < POSITION_TIMEOUT:
        for agent_id in [0, 1]:
            agent_name = f'agent_{agent_id}'
            pos = get_current_position(agent_id)
            if pos is not None:
                real_x, real_y, heading, grid_x, grid_y = pos
                agent_info[agent_name] = {
                    'real_pos': (real_x, real_y),
                    'grid_pos': (grid_x, grid_y),
                    'heading': heading,
                    'valid': True
                }
                if agent_name not in active_agents:
                    active_agents.append(agent_name)
                    print(f"[INFO] {agent_name} real: ({real_x:.1f},{real_y:.1f}) mm | "
                          f"grid: ({grid_x},{grid_y}) | heading: {heading:.2f}°")
            else:
                agent_info.setdefault(agent_name, {'valid': False})
        if active_agents:
            break
        time.sleep(POSITION_RETRY_INTERVAL)

    if goal_positions is None:
        goal_grid = [21, 20]
    else:
        goal_grid = goal_positions.get('agent_0', [21, 20])

    start_positions = {agent: (agent_info[agent]['grid_pos'] if agent_info[agent]['valid'] else [0, 0]) 
                       for agent in env.agents}

    obs, _ = env.reset(
        start_positions=[start_positions[agent] for agent in env.agents],
        goal_position=goal_grid
    )

    path_points = {agent: [start_positions[agent]] for agent in env.agents}
    for step in range(200):
        actions = maddpg.select_action(obs)
        obs, rewards, terminations, truncations, infos = env.step(actions)
        for agent in env.agents:
            pos = env.positions[agent].tolist()
            path_points[agent].append(pos)
        if all(terminations[agent] or truncations[agent] for agent in env.agents):
            break

    smoothed_paths = {agent: smooth_and_resample(path_points[agent], step=0.5, sigma=1.0, obstacles=obstacles)
                      for agent in env.agents}

    np.save("inferred_paths.npy", smoothed_paths)
    print("\n[INFO] Smoothed paths saved to inferred_paths.npy")

    # Visualization
    visualize_paths(smoothed_paths, obstacles)

    return smoothed_paths


if __name__ == "__main__":
    model_path = "./model 2.pt"
    agent_goals = {'agent_0': [21, 20], 'agent_1': [32, 31]}
    infer_path(model_path, agent_goals)

