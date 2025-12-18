import numpy as np
import torch
from src.environment import LoadBalancerEnv
from src.agents import DQNAgent

def run_trial(lr, gamma, use_dueling, episodes=100):
    """
    Runs a short training session (100 episodes) with specific parameters
    and returns the average reward of the last 20 episodes.
    """
    env = LoadBalancerEnv(num_servers=3)
    agent = DQNAgent(state_dim=3, action_dim=3, lr=lr, gamma=gamma, use_dueling=use_dueling) 
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
        
        if e % 10 == 0:
            agent.update_target_network()
            
        rewards.append(episode_reward)
    
    avg_score = np.mean(rewards[-20:])
    return avg_score

#Try different combinations sequentially.
def grid_search():
    learning_rates = [0.001, 0.0001]
    gammas = [0.99, 0.95]
    architectures = [False, True]
    
    best_score = -float('inf')
    best_params = {}
    
    print("Hyperparameter Tuning starting...") 
    for lr in learning_rates:
        for gamma in gammas:
            for use_dueling in architectures:
                
                score = run_trial(lr, gamma, use_dueling)
                
                print(f"{lr:<10} | {gamma:<10} | {str(use_dueling):<10} | {score:.2f}")
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        'lr': lr, 
                        'gamma': gamma, 
                        'use_dueling': use_dueling
                    }

    print("BEST RESULTS")
    print(f"Score: {best_score:.2f}")
    print(f"Parameters: {best_params}")

if __name__ == "__main__":
    grid_search()