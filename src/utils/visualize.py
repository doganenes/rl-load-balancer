import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import io 
import torch
import os 

from src.environment import LoadBalancerEnv
from src.agents import DQNAgent

def create_single_gif_in_memory(traffic_mode, filename, steps=50):
    print(f"\nPreparing simulation for '{traffic_mode.upper()}' traffic mode...")
    
    env = LoadBalancerEnv(num_servers=3)
    
    env.set_traffic_mode(traffic_mode)
    
    real_state_dim = env.observation_space.shape[0]
    agent = DQNAgent(state_dim=real_state_dim, action_dim=3, use_dueling=True)
    
    model_path = "dqn_load_balancer_6.pth"
    
    if os.path.exists(model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
        agent.policy_net.eval() 
        agent.epsilon = 0.0 
        print(f"Model loaded successfully: {model_path}")
    else:
        print("Error: Model not found!")

    state, _ = env.reset()
    frames = []

    all_loads_flat = []           
    server_loads = {0:[], 1:[], 2:[]} 
    requests_per_server = {0:0, 1:0, 2:0} 

    for step in range(steps):
        action = agent.select_action(state)
        next_state, reward, _, _, _ = env.step(action)
        
        for i in range(3):
            val = state[i]
            server_loads[i].append(val)
            all_loads_flat.append(val)
        
        requests_per_server[action] += 1
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        servers = ['Server 1', 'Server 2', 'Server 3']
        loads = state
        
        colors = []
        for load in loads:
            if load < 0.4: colors.append('#2ecc71') 
            elif load < 0.8: colors.append('#f1c40f') 
            elif load <= 1.0: colors.append('#e67e22')
            else: colors.append('#e74c3c')         
        
        bars = ax.bar(servers, loads, color=colors, edgecolor='black', alpha=0.9)
        
        bars[action].set_linewidth(3)
        bars[action].set_edgecolor('#3498db')
        
        current_max_load = np.max(loads)
        y_limit = max(1.25, current_max_load + 0.1)
        ax.set_ylim(0, y_limit)
        
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Max Capacity')
        
        mode_title = "LOW TRAFFIC" if traffic_mode == 'low' else "HIGH TRAFFIC"
        ax.set_title(f'{mode_title}\nStep: {step+1}/{steps} | Action: Sent to Server {action+1}', fontsize=10)
        ax.set_ylabel('CPU Load (%)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        
        buf = io.BytesIO()             
        plt.savefig(buf, format='png', dpi=100) 
        plt.close()                   
        buf.seek(0)                   
        
        frames.append(imageio.imread(buf))
        
        state = next_state

    imageio.mimsave(filename, frames, duration=0.5, loop=0)
    print(f"Saved GIF: {filename}")

    global_max = np.max(all_loads_flat)
    global_min = np.min(all_loads_flat)
    global_avg = np.mean(all_loads_flat)
    global_std = np.std(all_loads_flat)

    s1_avg = np.mean(server_loads[0])
    s2_avg = np.mean(server_loads[1])
    s3_avg = np.mean(server_loads[2])

    print(f"\nSTATS FOR {traffic_mode.upper()} TRAFFIC MODE")
    print(f"Global Max Load:      {global_max:.4f}")
    print(f"Global Min Load:      {global_min:.4f}")
    print(f"Global Average Load:  {global_avg:.4f}")
    print(f"Fairness (Std Dev):   {global_std:.4f}")
    print("-" * 45)
    print("Server Load Distribution:")
    print(f"  Server 1 -> Avg: {s1_avg:.4f} | Requests: {requests_per_server[0]}")
    print(f"  Server 2 -> Avg: {s2_avg:.4f} | Requests: {requests_per_server[1]}")
    print(f"  Server 3 -> Avg: {s3_avg:.4f} | Requests: {requests_per_server[2]}")
if __name__ == "__main__":
    if not os.path.exists("figures"):
        os.makedirs("figures")
        print("figures folder created.")

    path_low = os.path.join("figures", "simulation_low_traffic.gif")
    path_high = os.path.join("figures", "simulation_high_traffic.gif")

    create_single_gif_in_memory('low', path_low, steps=100)
    create_single_gif_in_memory('high', path_high, steps=100)
    