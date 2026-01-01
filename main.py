import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import logging
from src.environment import LoadBalancerEnv
from src.agents import DQNAgent

def train():
    """
    Trains a DQN agent for load balancing across multiple servers.
    Logs training progress, saves the trained model, and plots reward curves.
    """

    LR = 0.001
    GAMMA = 0.95
    USE_DUELING = True

    if not os.path.exists("logs"):
        os.makedirs("logs")

    log_path = os.path.join("logs", "training_curve.log")

    logging.basicConfig(
        filename=log_path,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s | %(message)s"
    )

    logging.info("Training started")
    logging.info(f"LR={LR}, Gamma={GAMMA}, Dueling={USE_DUELING}")

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
    print(f"{model_label} Main Training Starting... (LR={LR}, Gamma={GAMMA})")

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

        # ---- LOG PER EPISODE ----
        logging.info(
            f"Episode={e+1}, Reward={total_reward:.4f}, Epsilon={agent.epsilon:.4f}"
        )

        if (e + 1) % 50 == 0:
            print(
                f"Episode {e + 1}/{episodes} | "
                f"Reward: {total_reward:.2f} | "
                f"Epsilon: {agent.epsilon:.2f}"
            )

    print("Training Completed!")
    logging.info("Training completed")

    model_path = "models/dqn_load_balancer.pth"
    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"Model saved: {model_path}")
    logging.info(f"Model saved at {model_path}")

    if not os.path.exists("figures"):
        os.makedirs("figures")

    window = 50
    if len(rewards_history) >= window:
        smoothed_rewards = np.convolve(
            rewards_history,
            np.ones(window) / window,
            mode="valid"
        )
    else:
        smoothed_rewards = rewards_history

    for i, val in enumerate(smoothed_rewards):
        logging.info(
            f"MovingAvg_Episode={i + window}, Value={val:.4f}"
        )

    plt.figure(figsize=(10, 6))
    plt.plot(
        rewards_history,
        color='lightblue',
        alpha=0.4,
        label='Raw Rewards'
    )
    plt.plot(
        np.arange(len(smoothed_rewards)) + window - 1,
        smoothed_rewards,
        color='blue',
        linewidth=2.5,
        label='Moving Avg'
    )

    model_title = "Dueling DQN" if USE_DUELING else "Standard DQN"

    plt.title(
        f'{model_title} Training Performance\n(LR: {LR}, Gamma: {GAMMA})',
        fontsize=12,
        fontweight='bold'
    )
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join("figures", "training_curve.png")
    plt.savefig(plot_path)
    print(f"Training plot saved: {plot_path}")
    logging.info(f"Training curve saved at {plot_path}")

    plt.show()

if __name__ == "__main__":
    train()
