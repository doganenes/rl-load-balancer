import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import logging

from src.environment import LoadBalancerEnv
from src.agents import DQNAgent

"""
Compares Standard DQN and Dueling DQN architectures
under high traffic load using the same hyperparameters.
Generates a smoothed reward comparison plot.
"""

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/comparison_results.log",
    filemode="w",   
    level=logging.INFO, 
    format="%(asctime)s | %(message)s"
)

logger = logging.getLogger()

def set_seed(seed=42):
    """Ensures reproducibility by fixing random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info(f"Random seed set to {seed}")


def run_experiment(use_dueling, label, color, episodes, lr, gamma):
    """
    Trains a DQN-based agent and returns episode rewards.
    Used for fair architecture comparison.
    """
    print(f"Running Experiment: {label} (LR={lr}, Gamma={gamma})...")
    logger.info(f"Experiment started: {label} | LR={lr} | Gamma={gamma}")

    set_seed(42)

    env = LoadBalancerEnv(num_servers=3)
    env.set_traffic_mode('high')

    if hasattr(env, 'arrival_rate'):
        env.arrival_rate = 0.30
        logger.info(f"{label} | arrival_rate manually set to 0.30")

    state_dim = env.observation_space.shape[0]

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=3,
        lr=lr,
        gamma=gamma,
        use_dueling=use_dueling
    )

    agent.batch_size = 128
    agent.epsilon_decay = 0.995
    agent.epsilon_min = 0.01

    rewards = []

    for e in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0

        while not done and steps < 100:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.memory.push(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            episode_reward += reward
            steps += 1

        agent.update_epsilon()
        rewards.append(episode_reward)

        if (e + 1) % 50 == 0:
            avg_last_50 = np.mean(rewards[-50:])
            logger.info(
                f"{label} | Episode {e+1}/{episodes} | "
                f"AvgReward(Last50)={avg_last_50:.4f}"
            )

            print(
                f"  {label} | Episode {e+1}/{episodes} | "
                f"Avg (Last 50): {avg_last_50:.2f}"
            )

    logger.info(
        f"{label} | Training completed | "
        f"FinalAvg(Last50)={np.mean(rewards[-50:]):.4f}"
    )

    return rewards


def plot_rolling_stats(ax, rewards, label, color, window=50):
    """Plots episode rewards with rolling mean smoothing."""
    series = pd.Series(rewards)
    ax.plot(series, alpha=0.10, color=color)
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    x = np.arange(len(rewards))
    ax.plot(x, rolling_mean, label=f'{label}', color=color, linewidth=2.5)


def evaluate_architectures():
    """Runs both architectures and saves the comparison figure."""
    EPISODES = 800
    WINDOW = 50

    logger.info("Architecture Comparison Started")

    COMMON_LR = 0.0001
    COMMON_GAMMA = 0.95

    duel_rewards = run_experiment(
        use_dueling=True,
        label="Dueling DQN",
        color="red",
        episodes=EPISODES,
        lr=COMMON_LR,
        gamma=COMMON_GAMMA
    )

    std_rewards = run_experiment(
        use_dueling=False,
        label="Standard DQN",
        color="blue",
        episodes=EPISODES,
        lr=COMMON_LR,
        gamma=COMMON_GAMMA
    )
    
    print(f"\nDueling Rewards Count: {len(duel_rewards)}")
    print(f"Standard Rewards Count: {len(std_rewards)}")
    if len(duel_rewards) > 0:
        print(f"DEBUG: Sample Dueling Reward: {duel_rewards[-1]}")

    if not os.path.exists("figures"):
        os.makedirs("figures")
        logger.info("Figures directory created")

    fig, ax = plt.subplots(figsize=(12, 7))

    plot_rolling_stats(ax, std_rewards, f"Standard DQN (LR={COMMON_LR}, γ={COMMON_GAMMA})", "blue", window=WINDOW)
    plot_rolling_stats(ax, duel_rewards, f"Dueling DQN (LR={COMMON_LR}, γ={COMMON_GAMMA})", "red", window=WINDOW)

    ax.set_title(
        'Architecture Performance Comparison',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xlabel('Episodes', fontsize=12)
    ax.set_ylabel('Total Reward (Smoothed)', fontsize=12)

    ax.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    filename = 'compare_final_matched115.png'
    save_path = os.path.join("figures", filename)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot SAVED successfully to: {os.path.abspath(save_path)}")

    plt.close(fig)


    logger.info("Architecture Comparison Finished")
if __name__ == "__main__":
    evaluate_architectures()
