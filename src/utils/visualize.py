import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import io 
import torch
import os 
from src.environment import LoadBalancerEnv
from src.agents import DQNAgent

def create_single_gif_in_memory(traffic_mode, filename, steps=50):
    print(f"Preparing for '{traffic_mode.upper()}' traffic mode..")
    
    env = LoadBalancerEnv(num_servers=3)
    env.set_traffic_mode(traffic_mode)
    
    agent = DQNAgent(state_dim=3, action_dim=3)
    

    state, _ = env.reset()
    frames = []

    for step in range(steps):
        action = agent.select_action(state)
        next_state, reward, _, _, _ = env.step(action)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        servers = ['Server 1', 'Server 2', 'Server 3']
        loads = state
        
        colors = []
        for load in loads:
            if load < 0.4: colors.append('#2ecc71') 
            elif load < 0.8: colors.append('#f1c40f')
            else: colors.append('#e74c3c')           
        
        bars = ax.bar(servers, loads, color=colors, edgecolor='black', alpha=0.9)
        
        bars[action].set_linewidth(3)
        bars[action].set_edgecolor('#3498db')
        
        ax.set_ylim(0, 1.25)
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
    print(f"Saved: {filename}")

if __name__ == "__main__":
    if not os.path.exists("figures"):
        os.makedirs("figures")
        print("figures folder created.")

    path_low = os.path.join("figures", "simulation_low.gif")
    path_high = os.path.join("figures", "simulation_high.gif")

    create_single_gif_in_memory('low', path_low, steps=50)
    create_single_gif_in_memory('high', path_high, steps=50)
    
    print("PROCESS COMPLETED! Clean folder, only GIFs generated.")