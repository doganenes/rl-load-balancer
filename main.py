import numpy as np
import os   
import torch 
from src.environment import LoadBalancerEnv
from src.agents import DQNAgent
from src.utils.plot import plot_learning_curve

def train():
    env = LoadBalancerEnv(num_servers=3)
    
    agent = DQNAgent(
        state_dim=3, 
        action_dim=3, 
        lr=0.0001, 
        gamma=0.95, 
        use_dueling=True
    )
    
    episodes = 400  
    rewards_history = []
    
    print("DQN Main Training Starting...")
    
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
        
        if e % 10 == 0:
            print(f"Episode {e}/{episodes} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.2f}")

    print("Training Completed!")

    if not os.path.exists("figures"):
            os.makedirs("figures")
            print("'figures' folder created.")

    plot_path = os.path.join("figures", "training_curve.png")
    plot_learning_curve(rewards_history, filename=plot_path)
    print(f"Training plot saved: {plot_path}")

    if not os.path.exists("models"):
        os.makedirs("models")
        print("'models' folder created.")
    
    model_save_path = os.path.join("models", "dqn_load_balancer.pth")
    
    if os.path.exists(model_save_path):
        os.remove(model_save_path)
     
    torch.save(agent.policy_net.state_dict(), model_save_path)
    print(f"Model saved correctly to: {model_save_path}")

if __name__ == "__main__":
    train()