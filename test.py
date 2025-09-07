import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from environment.env_v1 import make_env
from maddpg.MADDPG import MADDPG
import random

def evaluate(env_id, maddpg, n_episode=10, episode_length=25):
    env = make_env(env_id)
    returns = np.zeros(len(env.agents))
    for _ in range(n_episode):
        obs = env.reset()
        for t_i in range(episode_length):
            actions = maddpg.select_action(obs)
            obs, rew, terminations, truncations, info = env.step(actions)
            done = {agent: terminations[agent] or truncations[agent] for agent in env.agents}

            rew = np.array(list(rew.values()))
            returns += rew / n_episode

            if all(done.values()):
                break
    return returns.tolist()

# 画累计奖励曲线
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
    # trajectory_list 是一个列表，每个元素是一个字典 {agent: np.array of shape (episode_length, 2)}
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
    # 超参数
    env_id = "GridWorld-v0"
    num_episodes = 10000
    episode_length = 25
    batch_size = 1024
    minimal_size = 2000
    buffer_capacity = 100000
    update_interval = 100
    eval_interval = 100
    gamma = 0.95
    tau = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("The device is: ", device)

    # 创建结果目录
    result_dir = "./.results"
    os.makedirs(result_dir, exist_ok=True)

    # 创建环境和MADDPG实例
    env = make_env(env_id)
    n_agents = len(env.agents)
    dim_info = {agent: (env.observation_spaces[agent].shape[0], env.action_spaces[agent].n) for agent in env.agents}
    maddpg = MADDPG(dim_info, buffer_capacity, batch_size, actor_lr=1e-2, critic_lr=1e-2, res_dir=result_dir)

    # 训练主循环
    return_list = []
    trajectory_list = []  # 记录轨迹

    total_step = 0
    for i_episode in range(num_episodes):
        obs = env.reset()

        # 每个 agent 轨迹记录
        episode_positions = {agent: [] for agent in env.agents}
        episode_rewards = {agent: 0 for agent in env.agents}

        for t in range(episode_length):
            actions = maddpg.select_action(obs)
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            dones = {agent: terminations[agent] or truncations[agent] for agent in env.agents}

            # 记录位置和奖励
            for agent in env.agents:
                episode_positions[agent].append(env.positions[agent].copy())
                episode_rewards[agent] += rewards[agent]

            maddpg.add(obs, actions, rewards, next_obs, dones)
            obs = next_obs
            total_step += 1

            if all(len(buffer) >= batch_size for buffer in maddpg.buffers.values()) and total_step % update_interval == 0:
                maddpg.learn(batch_size, gamma)
                maddpg.update_target(tau)

        # episode 结束，保存轨迹和累计奖励
        trajectory_list.append({agent: np.array(pos_list) for agent, pos_list in episode_positions.items()})
        return_list.append([episode_rewards[agent] for agent in env.agents])

        if (i_episode + 1) % eval_interval == 0:
            ep_returns = evaluate(env_id, maddpg, n_episode=10)
            print(f"Episode {i_episode + 1}: Average Return: {ep_returns}")

            # 保存奖励曲线和轨迹图
            plot_returns(return_list)
            plot_trajectories(trajectory_list)

            # 保存模型
            maddpg.save(return_list)
            # 你还可以把log写入文件，示范写log
            with open(os.path.join(result_dir, "maddpg.log"), "a") as f_log:
                f_log.write(f"Episode {i_episode + 1}: Average Return: {ep_returns}\n")