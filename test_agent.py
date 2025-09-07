import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from environment.env_v3 import make_env
from maddpg.MADDPG import MADDPG

def visualize_episode(env, maddpg, episode_length=25, save_path=None):
    """Visualize a full episode and save the trajectory plot"""
    obs = env.reset()
    positions = {agent: [] for agent in env.agents}
    rewards = {agent: 0 for agent in env.agents}
    
    plt.figure(figsize=(8, 8))
    
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
    
    # Plot trajectories
    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    for i, agent in enumerate(env.agents):
        traj = np.array(positions[agent])
        plt.plot(traj[:, 0], traj[:, 1], color=colors[i % len(colors)], 
                marker='o', markersize=4, label=f'{agent} (R:{rewards[agent]:.1f})')
        plt.scatter(traj[0, 0], traj[0, 1], color=colors[i % len(colors)], 
                   marker='s', s=100, edgecolor='k')  # Start point
        plt.scatter(traj[-1, 0], traj[-1, 1], color=colors[i % len(colors)], 
                   marker='*', s=200, edgecolor='k')  # End point
    
    plt.title('Agent Trajectories')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Trajectory plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return rewards

def evaluate_model(env_id, maddpg, n_episode=10, episode_length=25):
    """Evaluate model performance and return average rewards"""
    env = make_env(env_id)
    total_rewards = {agent: 0 for agent in env.agents}
    
    for _ in range(n_episode):
        obs = env.reset()
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
    
    return {agent: total_rewards[agent]/n_episode for agent in env.agents}

if __name__ == "__main__":
    # Configuration
    env_id = "GridWorld-v0"
    model_path = "./results_test_20250825_164004/seed_2/model.pt"  # model path
    test_episodes = 5                  # number of test episodes
    episode_length = 25                # max steps per episode
    
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
        print("2. Whether the environment configuration matches the training")
        exit()
    
    # Create directory for test results
    test_dir = os.path.join(os.path.dirname(model_path), "test_results1")
    os.makedirs(test_dir, exist_ok=True)
    
    # Evaluate model performance
    print("\nEvaluating model performance...")
    avg_rewards = evaluate_model(env_id, maddpg)
    print("Average rewards per agent:")
    for agent, reward in avg_rewards.items():
        print(f"{agent}: {reward:.2f}")
    
    # Visualize trajectories
    print("\nGenerating trajectory visualizations...")
    for i in range(test_episodes):
        print(f"\nTest Episode {i+1}:")
        save_path = os.path.join(test_dir, f"trajectory_ep{i+1}.png")
        rewards = visualize_episode(env, maddpg, episode_length, save_path)
        print("Episode rewards:")
        for agent, reward in rewards.items():
            print(f"{agent}: {reward:.2f}")
