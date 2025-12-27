import numpy as np
import torch
import pandas as pd
from joblib import Parallel, delayed
from src.environment import LoadBalancerEnv
from src.agents import DQNAgent

def train_single_seed(seed, lr, gamma, use_dueling, episodes):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = LoadBalancerEnv(num_servers=3)
    env.set_traffic_mode("low") 

    
    agent = DQNAgent(
        state_dim=3,
        action_dim=3,
        lr=lr,
        gamma=gamma,
        use_dueling=use_dueling
    )

    rewards = []

    for e in range(episodes):
        state, _ = env.reset(seed=seed+e) 
        episode_reward = 0
        steps = 0
        done = False

        while not done and steps < 100:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            agent.memory.push(state, action, reward, next_state, terminated)
            
            if len(agent.memory) > 32 and steps % 4 == 0:
                agent.learn()

            if steps % 10 == 0:
                agent.update_target_network()

            state = next_state
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        agent.update_epsilon()
        rewards.append(episode_reward)

    return np.mean(rewards[-20:])

def run_trial_parallel(lr, gamma, use_dueling, episodes=800):
    seeds = [1, 2, 3]
    
    results = Parallel(n_jobs=-1)(
        delayed(train_single_seed)(seed, lr, gamma, use_dueling, episodes) 
        for seed in seeds
    )
    
    return np.mean(results)

def grid_search():
    learning_rates = [0.001, 0.0005, 0.0001]
    gammas = [0.99, 0.95, 0.90]
    architectures = [False, True]
   
    results = []
   
    print(f"Grid Search Started..")
    print("-" * 75)
    print(f"{'LR':<10} | {'Gamma':<10} | {'Dueling':<10} | {'Avg Score':<15}")
    print("-" * 75)

    for lr in learning_rates:
        for gamma in gammas:
            for use_dueling in architectures:
                
                score = run_trial_parallel(lr, gamma, use_dueling)
                
                print(f"{lr:<10} | {gamma:<10} | {str(use_dueling):<10} | {score:.2f}")
                
                results.append({
                    'Learning Rate': lr,
                    'Gamma': gamma,
                    'Architecture': 'Dueling DQN' if use_dueling else 'Standard DQN',
                    'Average Reward': score
                })

    print("-" * 75)
   
    df = pd.DataFrame(results)
    df = df.sort_values(by='Average Reward', ascending=False)
    df.insert(0, 'Rank', range(1, 1 + len(df)))
   
    filename = 'tuning_results_parallel.csv'
    df.to_csv(filename, index=False)
    print(f"\nResults saved to '{filename}'.")
    print(df.head(10))

if __name__ == "__main__":
    grid_search()