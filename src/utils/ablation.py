import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from src.environment import LoadBalancerEnv
from src.agents import DQNAgent

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_experiment(use_dueling, label, color, lr, episodes=400):
    print(f"Experiment: {label}")
    set_seed(42)
    env = LoadBalancerEnv(num_servers=3)
    
    agent = DQNAgent(state_dim=3, action_dim=3, lr=lr, gamma=0.9, use_dueling=use_dueling)
    agent.batch_size = 256
    
    rewards = []
    
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        while not done and steps < 100:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward
            steps += 1
            
        agent.update_epsilon()
        rewards.append(total_reward)

    window = 15
    if len(rewards) >= window:
        smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    else:
        smoothed_rewards = rewards
    return smoothed_rewards

def run_ablation_study():
    print("------------------------------------------------")
    print("ABLATION STUDY: Standard DQN vs Dueling DQN")
    print("------------------------------------------------")
    
    std_rewards = run_experiment(use_dueling=False, label="Standard DQN", color="blue", lr=0.0001)
    duel_rewards = run_experiment(use_dueling=True, label="Dueling DQN", color="red", lr=0.0001)
    
    if not os.path.exists("figures"):
        os.makedirs("figures")

    plt.figure(figsize=(10, 6))
    plt.plot(std_rewards, label='Standard DQN', color='blue', alpha=0.6, linestyle='--')
    plt.plot(duel_rewards, label='Dueling DQN (Advanced)', color='red', linewidth=2.5)
    
    plt.title('Ablation Study: Architecture Impact\n(Standard vs Dueling DQN)')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward (Higher is Better)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = 'ablation_final_fixed.png'
    save_path = os.path.join("figures", filename)
    plt.savefig(save_path)
    
    print(f"\nPlot saved successfully: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_ablation_study()
