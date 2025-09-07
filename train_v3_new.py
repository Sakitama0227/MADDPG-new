import os  
import numpy as np
import torch
import matplotlib.pyplot as plt
from environment.env_v3 import make_env
from maddpg.MADDPG import MADDPG
import random
import json
from datetime import datetime
import pandas as pd
from tqdm import tqdm

def set_seed(seed):
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate(env, maddpg, n_episode=10, episode_length=100):
    """Evaluation function, returns average reward and success rate"""
    returns = np.zeros(len(env.agents))
    success_count = 0
    
    for _ in range(n_episode):
        obs, _ = env.reset()
        episode_reward = {agent: 0 for agent in env.agents}
        
        for t_i in range(episode_length):
            actions = maddpg.select_action(obs)
            obs, rew, terminations, truncations, info = env.step(actions)
            done = {agent: terminations[agent] or truncations[agent] for agent in env.agents}

            for agent in env.agents:
                episode_reward[agent] += rew[agent]
            
            if all(terminations.values()):  # Successfully reached goal
                success_count += 1
                break
            if all(done.values()):  # Timeout or other termination
                break
        
        returns += np.array([episode_reward[agent] for agent in env.agents]) / n_episode
    
    success_rate = success_count / n_episode
    return returns.tolist(), success_rate

def plot_training_curves(all_results, result_dir, downsample_interval=200, smooth_window=5):
    """Plot multi-seed training curves + average curve + std shading (with downsampling and smoothing)"""
    if not all_results:
        print("No results to plot")
        return
        
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Prepare data
    seeds = list(all_results.keys())
    colors = ['blue', 'red', 'green', 'orange', 'purple', 
              'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Define metrics to plot
    metrics = {
        'episode_rewards': ("Total Reward", 'Episodes', 'Reward'),
        'success_rates': ("Success Rate", 'Episodes', 'Success Rate'),
        'episode_steps': ("Episode Length", 'Episodes', 'Steps'),
        'mean_distances': ("Mean Distance to Goal", 'Episodes', 'Distance'),
        'collision_counts': ("Collision Count", 'Episodes', 'Collisions')
    }
    
    axes_list = [axes[0,0], axes[0,1], axes[0,2], axes[1,0], axes[1,1]]
    
    def downsample(values, interval=200):
        """Average every 'interval' episodes to reduce points"""
        return [np.mean(values[i:i+interval]) for i in range(0, len(values), interval)]
    
    for (metric, (title, xlabel, ylabel)), ax in zip(metrics.items(), axes_list):
        all_curves = []
        x_axis = None
        
        for seed, color in zip(seeds, colors[:len(seeds)]):
            values = all_results[seed][metric]
            
            # Step 1: Downsample
            ds_values = downsample(values, interval=downsample_interval)
            # Step 2: Rolling smoothing
            smoothed = pd.Series(ds_values).rolling(window=smooth_window, min_periods=1).mean().values
            
            x_axis = range(0, len(values), downsample_interval)
            ax.plot(x_axis, smoothed, label=f'Seed {seed}', color=color, alpha=0.6)
            all_curves.append(smoothed)
        
        # Align lengths
        min_len = min(len(c) for c in all_curves)
        trimmed = [c[:min_len] for c in all_curves]
        stacked = np.vstack(trimmed)
        
        # Mean and std
        mean_curve = np.mean(stacked, axis=0)
        std_curve = np.std(stacked, axis=0)
        
        # Plot average curve + shading
        ax.plot(x_axis[:min_len], mean_curve, label='Average', color='black', linewidth=2.0)
        ax.fill_between(x_axis[:min_len], 
                        mean_curve - std_curve, 
                        mean_curve + std_curve, 
                        color='gray', alpha=0.2, label='±1 Std')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Confidence interval plot (average performance of last 100 episodes)
    ax = axes[1, 2]
    final_performance = []
    for seed in seeds:
        final_rewards = all_results[seed]['episode_rewards'][-100:]
        final_performance.append(np.mean(final_rewards))
    
    mean_perf = np.mean(final_performance)
    std_perf = np.std(final_performance)
    ci = 1.96 * std_perf / np.sqrt(len(seeds))
    
    ax.bar(['Final Performance'], [mean_perf], yerr=[ci], capsize=10, 
           color='skyblue', alpha=0.7)
    ax.set_ylabel('Average Reward')
    ax.set_title(f'Final Performance (95% CI)\nMean: {mean_perf:.2f} ± {ci:.2f}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "training_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()


def save_statistical_analysis(all_results, result_dir):
    """Save statistical analysis results"""
    if not all_results:
        return {}
        
    analysis_results = {}
    
    # Compute final performance for each seed
    final_performances = {}
    for seed, results in all_results.items():
        final_100 = results['episode_rewards'][-100:]
        final_performances[seed] = {
            'mean_reward': np.mean(final_100),
            'std_reward': np.std(final_100),
            'success_rate': np.mean(results['success_rates'][-100:]),
            'mean_steps': np.mean(results['episode_steps'][-100:]),
            'mean_distance': np.mean(results['mean_distances'][-100:]),
            'mean_collisions': np.mean(results['collision_counts'][-100:])
        }
    
    # Compute overall statistics
    all_rewards = [perf['mean_reward'] for perf in final_performances.values()]
    all_success = [perf['success_rate'] for perf in final_performances.values()]
    
    analysis_results['overall'] = {
        'mean_final_reward': np.mean(all_rewards),
        'std_final_reward': np.std(all_rewards),
        'mean_final_success': np.mean(all_success),
        'std_final_success': np.std(all_success),
        'confidence_interval_reward': 1.96 * np.std(all_rewards) / np.sqrt(len(all_rewards)),
        'confidence_interval_success': 1.96 * np.std(all_success) / np.sqrt(len(all_success)),
        'min_reward': np.min(all_rewards),
        'max_reward': np.max(all_rewards),
        'min_success': np.min(all_success),
        'max_success': np.max(all_success)
    }
    
    analysis_results['seed_details'] = final_performances
    
    # Save to file
    with open(os.path.join(result_dir, "statistical_analysis.json"), 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    return analysis_results

def run_single_seed(seed, config, result_dir):
    """Run training for a single seed"""
    print(f"Starting training for seed {seed}")
    set_seed(seed)
    
    # Create environment
    env = make_env(config['env_id'], seed=seed, enable_monitoring=True)
    n_agents = len(env.agents)
    
    # Create MADDPG instance
    dim_info = {agent: (env.observation_spaces[agent].shape[0], env.action_spaces[agent].n) 
               for agent in env.agents}
    maddpg = MADDPG(dim_info, config['buffer_capacity'], config['batch_size'],
                   actor_lr=config['actor_lr'], critic_lr=config['critic_lr'],
                   res_dir=os.path.join(result_dir, f"seed_{seed}"))
    
    results = {
        'episode_rewards': [],
        'success_rates': [],
        'episode_steps': [],
        'mean_distances': [],
        'collision_counts': [],
        'eval_returns': [],
        'eval_success_rates': []
    }
    
    total_step = 0
    best_success_rate = 0.0
    best_model_path = None
    
    for i_episode in tqdm(range(config['num_episodes']), desc=f"Seed {seed}"):
        obs, _ = env.reset()
        episode_reward = 0
        
        for t in range(config['episode_length']):
            actions = maddpg.select_action(obs)
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            dones = {agent: terminations[agent] or truncations[agent] for agent in env.agents}
            
            # Record the total reward
            episode_reward += sum(rewards.values())
            
            # Add to experience replay
            maddpg.add(obs, actions, rewards, next_obs, dones)
            obs = next_obs
            total_step += 1
           
            # Train if enough samples and it's update interval
            if all(len(buffer) >= config['batch_size'] for buffer in maddpg.buffers.values()) and total_step % config['update_interval'] == 0:
                maddpg.learn(config['batch_size'], config['gamma'])
                maddpg.update_target(config['tau'])
            
            if all(dones.values()):
                break
        
        episode_summary = env.record_episode_to_training_monitor()
        results['episode_rewards'].append(episode_summary['total_reward'])
        results['success_rates'].append(episode_summary['success_rate'])
        results['episode_steps'].append(episode_summary['steps'])
        results['mean_distances'].append(episode_summary['mean_distance'])
        results['collision_counts'].append(episode_summary['total_collisions'])
        
        # Regular evaluation
        if (i_episode + 1) % config['eval_interval'] == 0:
            eval_returns, eval_success = evaluate(env, maddpg, n_episode=5)
            results['eval_returns'].append(eval_returns)
            results['eval_success_rates'].append(eval_success)
            
            print(f"Seed {seed}, Episode {i_episode + 1}: "
                  f"Reward: {episode_summary['total_reward']:.2f}, "
                  f"Success: {episode_summary['success_rate']:.2f}, "
                  f"Eval Success: {eval_success:.2f}")
            
            # Save the best model
            if eval_success > best_success_rate:
                best_success_rate = eval_success
                current_reward = episode_summary['total_reward']
                maddpg.save(current_reward)  
        
        # Save intermediate training data every 1000 episodes
        if (i_episode + 1) % 1000 == 0:
            env.save_training_data(os.path.join(result_dir, f"seed_{seed}", f"training_data_ep_{i_episode+1}.json"))
    
    # Save final model and training data
    final_reward = results['episode_rewards'][-1] if results['episode_rewards'] else 0
    maddpg.save(final_reward)  
    env.save_training_data(os.path.join(result_dir, f"seed_{seed}", "training_data_final.json"))
    
    return results
