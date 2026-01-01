import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import logging

from src.environment import LoadBalancerEnv
from src.agents import DQNAgent

"""
Performs an ablation study on a DQN-based load balancing agent.

This provides:
- Compares different DQN variants (Standard, Dueling, without target network, without replay memory).
- Trains each variant in a LoadBalancer environment under specified traffic conditions.
- Logs rewards and training progress.
- Plots and saves smoothed reward curves for each experiment.
"""

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/ablation_results.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)
logger = logging.getLogger()

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_experiment(label, episodes, lr, gamma, use_dueling=True, use_target=True, use_replay=True, traffic_mode="low"):
    # Runs a single training experiment with specified DQN configuration.
    logger.info(f"Starting: {label} | Traffic: {traffic_mode}")
    set_seed(42)

    env = LoadBalancerEnv(num_servers=3)
    env.set_traffic_mode(traffic_mode)

    agent = DQNAgent(
        state_dim=3, 
        action_dim=3, 
        lr=lr, 
        gamma=gamma, 
        use_dueling=use_dueling,
        use_target_network=use_target,
        use_replay_memory=use_replay
    )

    rewards_history = []

    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        while not done and step_count < 100:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            
            agent.learn()

            state = next_state
            total_reward += reward
            step_count += 1

        agent.update_epsilon()
        rewards_history.append(total_reward)

        if (e + 1) % 50 == 0:
            avg_last_50 = np.mean(rewards_history[-50:])
            print(f"{label} ({traffic_mode}) | Episode {e+1}/{episodes} | Avg Reward: {avg_last_50:.2f} | Eps: {agent.epsilon:.2f}")

    return rewards_history

def plot_ablation_results(all_results, traffic_mode):
    """Plots smoothed reward curves for multiple experiments."""

    plt.figure(figsize=(10, 6))
    window = 50
    
    for label, rewards, color in all_results:
        series = pd.Series(rewards)
        smoothed = series.rolling(window=window, min_periods=1).mean()
        
        plt.plot(smoothed, label=label, color=color, linewidth=2.5)

    plt.title(f'Ablation Study: {traffic_mode.upper()} Traffic Performance', fontsize=12, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/ablation_{traffic_mode}_traffic.png")
    plt.close()

def run_ablation_study():
    """Executes all ablation experiments and saves figures."""
    EPISODES = 800
    LR = 0.001    
    GAMMA = 0.99  
    
    experiments = [
        {"label": "Standard DQN", "dueling": False, "target": True, "replay": True, "color": "blue"},
        {"label": "Dueling DQN", "dueling": True, "target": True, "replay": True, "color": "red"},
        {"label": "No Target Network", "dueling": True, "target": False, "replay": True, "color": "green"},
        {"label": "No Replay Memory", "dueling": True, "target": True, "replay": False, "color": "orange"}
    ]

    for traffic_mode in ["high"]:
        print(f"\n--- Starting {traffic_mode.upper()} Traffic Experiments ---")
        traffic_results = []
        
        for exp in experiments:
            rewards = run_experiment(
                label=exp['label'],
                episodes=EPISODES,
                lr=LR,
                gamma=GAMMA,
                use_dueling=exp["dueling"],
                use_target=exp["target"],
                use_replay=exp["replay"],
                traffic_mode=traffic_mode
            )
            traffic_results.append((exp['label'], rewards, exp['color']))
        
        plot_ablation_results(traffic_results, traffic_mode)

if __name__ == "__main__":
    run_ablation_study()