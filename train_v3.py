import os 
import numpy as np
import torch
import matplotlib.pyplot as plt
from environment.env_v3 import make_env
from maddpg.MADDPG import MADDPG
import random

def evaluate(env_id, maddpg, n_episode=10, episode_length=100):
    env = make_env(env_id)
    returns = np.zeros(len(env.agents))
    for _ in range(n_episode):
        obs, _ = env.reset()  
        for t_i in range(episode_length):
            actions = maddpg.select_action(obs)
            obs, rew, terminations, truncations, info = env.step(actions)
            done = {agent: terminations[agent] or truncations[agent] for agent in env.agents}

            rew = np.array(list(rew.values()))
            returns += rew / n_episode

            if all(done.values()):
                break
    return returns.tolist()


def plot_returns(return_list):
    plt.figure()
    return_array = np.array(return_list)  # shape: episodes x agents
    for agent_i in range(return_array.shape[1]):
        plt.plot(return_array[:, agent_i], label=f"Agent {agent_i}")
    plt.xlabel("Evaluation Episodes (x eval_interval)")
    plt.ylabel("Average Return")
    plt.title("Average Return per Agent")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(result_dir, "return_curve.png"))
    plt.close()

# 画轨迹
def plot_trajectories(trajectory_list):
    
    plt.figure(figsize=(8, 8))
    colors = ['r', 'g', 'b', 'y', 'm', 'c']

    for agent_i in range(n_agents):
        for ep_i, traj_dict in enumerate(trajectory_list):
            traj = traj_dict[env.agents[agent_i]]  # shape (episode_length, 2)
            x = traj[:, 0]
            y = traj[:, 1]
            plt.plot(x, y, color=colors[agent_i % len(colors)], alpha=0.3)
        plt.scatter([], [], color=colors[agent_i % len(colors)], label=f"Agent {agent_i}")

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Agent Trajectories")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(result_dir, "trajectories.png"))
    plt.close()

if __name__ == "__main__":
    
    env_id = "GridWorld_v1"
    num_episodes = 15000  
    episode_length = 100  
    
    batch_size = 512      
    minimal_size = 5000   
    buffer_capacity = int(1e6)  
    
    # Training Parameters
    update_interval = 50 
    eval_interval = 500   
    gamma = 0.99          
    tau = 0.005           
    
    # learning rate
    actor_lr = 1e-4       
    critic_lr = 1e-3      
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    result_dir = "./.results"
    os.makedirs(result_dir, exist_ok=True)

    # Create environments and MADDPG instances
    env = make_env(env_id)
    n_agents = len(env.agents)
    dim_info = {agent: (env.observation_spaces[agent].shape[0], env.action_spaces[agent].n) for agent in env.agents}
    maddpg = MADDPG(dim_info, buffer_capacity, batch_size, actor_lr=actor_lr, critic_lr=critic_lr, res_dir=result_dir)


    # Training the main loop
    return_list = []
    trajectory_list = []  

    total_step = 0
    for i_episode in range(num_episodes):
        obs, _ = env.reset()


        episode_positions = {agent: [] for agent in env.agents}
        episode_rewards = {agent: 0 for agent in env.agents}

        for t in range(episode_length):
            actions = maddpg.select_action(obs)
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            dones = {agent: terminations[agent] or truncations[agent] for agent in env.agents}

            for agent in env.agents:
                episode_positions[agent].append(env.positions[agent].copy())
                episode_rewards[agent] += rewards[agent]

            maddpg.add(obs, actions, rewards, next_obs, dones)
            obs = next_obs
            total_step += 1

            if all(len(buffer) >= batch_size for buffer in maddpg.buffers.values()) and total_step % update_interval == 0:
                maddpg.learn(batch_size, gamma)
                maddpg.update_target(tau)

        
        return_list.append([episode_rewards[agent] for agent in env.agents])

        if (i_episode + 1) % eval_interval == 0:
            ep_returns = evaluate(env_id, maddpg, n_episode=10)
            print(f"Episode {i_episode + 1}: Average Return: {ep_returns}")

            plot_returns(return_list)
      
            maddpg.save(return_list)
        
        if i_episode == 0:
            with open(os.path.join(result_dir, "maddpg.log"), "w") as f_log:
                f_log.write("MADDPG Training Log\n")