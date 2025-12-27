import numpy as np
import os
import torch
import matplotlib.pyplot as plt

from src.environment import LoadBalancerEnv
from src.agents import DQNAgent


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, "train.log")

def log_print(message):
    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def train():

    LR = 0.001
    GAMMA = 0.95
    USE_DUELING = True

    env = LoadBalancerEnv(num_servers=3)
    agent = DQNAgent(
        state_dim=3,
        action_dim=3,
        lr=LR,
        use_dueling=USE_DUELING,
        gamma=GAMMA
    )

    episodes = 800
    rewards_history = []

    model_label = "Dueling DQN" if USE_DUELING else "Standard DQN"
    log_print(f"{model_label} Main Training Starting... (LR={LR}, Gamma={GAMMA})")
    log_print(f"Log file: {log_file}")
    log_print("-" * 60)

    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        while not done and step_count < 100:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.memory.push(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward
            step_count += 1

        agent.update_epsilon()
        rewards_history.append(total_reward)

        if (e + 1) % 50 == 0:
            log_print(
                f"Episode {e + 1}/{episodes} | "
                f"Reward: {total_reward:.2f} | "
                f"Epsilon: {agent.epsilon:.2f}"
            )

    log_print("Training Completed!")
    log_print("-" * 60)

    model_path = "dqn_load_balancer_6.pth"
    torch.save(agent.policy_net.state_dict(), model_path)
    log_print(f"Model saved: {model_path}")

    if not os.path.exists("figures"):
        os.makedirs("figures")

    window = 50
    if len(rewards_history) >= window:
        smoothed_rewards = np.convolve(
            rewards_history,
            np.ones(window) / window,
            mode='valid'
        )
    else:
        smoothed_rewards = rewards_history

    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history, color='lightblue', alpha=0.4, label='Raw Rewards')
    plt.plot(
        np.arange(len(smoothed_rewards)) + window - 1,
        smoothed_rewards,
        color='blue',
        linewidth=2.5,
        label='Moving Avg'
    )

    plt.title(
        f'{model_label} Training Performance\n(LR: {LR}, Gamma: {GAMMA})',
        fontsize=12,
        fontweight='bold'
    )
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join("figures", "training_curve_final_6.png")
    plt.savefig(plot_path)
    log_print(f"Training plot saved: {plot_path}")
    plt.show()


if __name__ == "__main__":
    train()
