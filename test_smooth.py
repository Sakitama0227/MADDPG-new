import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from environment.env_v3 import make_env
from maddpg.MADDPG import MADDPG


# ------------------ Smoothing Utility Functions ------------------
def bresenham_line(x0, y0, x1, y1):
    """Generate grid coordinates between two points using Bresenham's algorithm"""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points


def line_of_sight(p1, p2, obstacles):
    """Check if a straight line between two points is free of obstacles"""
    for x, y in bresenham_line(p1[0], p1[1], p2[0], p2[1]):
        if obstacles[y, x]:  # obstacles is indexed by [y,x]
            return False
    return True


def smooth_path(path, obstacles):
    """Smooth a path using line-of-sight checks"""
    if len(path) <= 2:
        return path
    smooth = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if line_of_sight(path[i], path[j], obstacles):
                smooth.append(path[j])
                i = j
                break
            j -= 1
        else:  # If no direct line found, keep the next point
            smooth.append(path[i + 1])
            i += 1
    return smooth


# ------------------ Visualization Functions ------------------
def visualize_episode(env, maddpg, episode_length=200, save_path=None):
    """Visualize a full episode and save the trajectory plot (full 40x40 map)"""
    obs, _ = env.reset()
    positions = {agent: [] for agent in env.agents}
    rewards = {agent: 0 for agent in env.agents}

    grid_size = getattr(env, "grid_size", 40)  # default to 40 if not defined
    plt.figure(figsize=(8, 8))

    # Run a full episode
    for t in range(episode_length):
        actions = maddpg.select_action(obs)
        next_obs, rew, terminations, truncations, _ = env.step(actions)
        done = {agent: terminations[agent] or truncations[agent] for agent in env.agents}

        for agent in env.agents:
            positions[agent].append(env.positions[agent].copy())
            rewards[agent] += rew[agent]

        obs = next_obs
        if all(done.values()):
            break

    # Fix axis to display the whole map
    plt.xlim(-0.5, grid_size - 0.5)
    plt.ylim(-0.5, grid_size - 0.5)
    plt.gca().set_aspect('equal')  # Keep aspect ratio

    # Invert y-axis so 0 is top and 39 is bottom
    plt.gca().invert_yaxis()

    # Draw background grid
    plt.xticks(range(grid_size))
    plt.yticks(range(grid_size))
    plt.grid(True, which='both', color='lightgray', linewidth=0.5)

    # Draw obstacles
    obs_map = None
    if hasattr(env, "obstacles"):
        obs_map = env.obstacles.astype(int)
    else:
        obstacle_file = os.path.join(os.path.dirname(__file__), "obstacle_map.txt")
        if os.path.exists(obstacle_file):
            obs_map = np.loadtxt(obstacle_file, dtype=int)

    if obs_map is not None:
        for y in range(obs_map.shape[0]):
            for x in range(obs_map.shape[1]):
                if obs_map[y, x] == 1:  # obstacle
                    plt.scatter(x, y, color='k', marker='s', s=100)

    # Draw shared goal
    goal = env.goal
    plt.scatter(goal[0], goal[1], color='k', marker='X', s=200, label='Goal')

    # Draw trajectories (with smoothing)
    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    for i, agent in enumerate(env.agents):
        traj = np.array(positions[agent])
        # Smooth trajectory only if obstacle map exists
        if obs_map is not None:
            traj = np.array(smooth_path([tuple(p) for p in traj], obs_map))
        plt.plot(traj[:, 0], traj[:, 1],
                 color=colors[i % len(colors)],
                 marker='o', markersize=4,
                 label=f'{agent} (R:{rewards[agent]:.1f})')
        plt.scatter(traj[0, 0], traj[0, 1],
                    color=colors[i % len(colors)],
                    marker='s', s=100, edgecolor='k')  # start
        plt.scatter(traj[-1, 0], traj[-1, 1],
                    color=colors[i % len(colors)],
                    marker='*', s=200, edgecolor='k')  # end

    plt.title('Agent Trajectories (Full 40x40 Map, Smoothed)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Trajectory plot saved to {save_path}")
    else:
        plt.show()

    plt.close()
    return rewards


def evaluate_model(env, maddpg, n_episode=10, episode_length=500):
    """Evaluate model performance and return average rewards"""
    total_rewards = {agent: 0 for agent in env.agents}

    for _ in range(n_episode):
        obs, _ = env.reset()
        episode_rewards = {agent: 0 for agent in env.agents}

        for t in range(episode_length):
            actions = maddpg.select_action(obs)
            next_obs, rew, terminations, truncations, _ = env.step(actions)
            done = {agent: terminations[agent] or truncations[agent] for agent in env.agents}

            for agent in env.agents:
                episode_rewards[agent] += rew[agent]

            obs = next_obs
            if all(done.values()):
                break

        for agent in env.agents:
            total_rewards[agent] += episode_rewards[agent]

    return {agent: total_rewards[agent] / n_episode for agent in env.agents}


if __name__ == "__main__":
    # Configuration
    env_id = "GridWorld-v1"
    model_path = "./results_test_20250825_164004/seed_2/model.pt"  # model path
    test_episodes = 5                  # number of test episodes
    episode_length = 200               # max steps per episode

    # Create environment to get dimension info
    env = make_env(env_id)
    dim_info = {agent: (env.observation_spaces[agent].shape[0],
                        env.action_spaces[agent].n) for agent in env.agents}

    # Load trained model
    print("Loading trained model...")
    try:
        maddpg = MADDPG.load(dim_info, model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Please check:")
        print(f"1. Whether the model file exists: {model_path}")
        print("2. Whether the environment configuration matches training")
        exit()

    # Create directory for test results
    test_dir = os.path.join(os.path.dirname(model_path), "test_results")
    os.makedirs(test_dir, exist_ok=True)

    # Evaluate model performance
    print("\nEvaluating model performance...")
    avg_rewards = evaluate_model(env, maddpg)  # pass the same env instance
    print("Average rewards per agent:")
    for agent, reward in avg_rewards.items():
        print(f"{agent}: {reward:.2f}")

    # Visualize trajectories
    print("\nGenerating trajectory visualizations...")
    for i in range(test_episodes):
        print(f"\nTest Episode {i + 1}:")
        save_path = os.path.join(test_dir, f"trajectory_ep{i + 1}.png")
        rewards = visualize_episode(env, maddpg, episode_length, save_path)
        print("Episode rewards:")
        for agent, reward in rewards.items():
            print(f"{agent}: {reward:.2f}")
