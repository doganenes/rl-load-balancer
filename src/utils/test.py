import numpy as np
import matplotlib.pyplot as plt
import os 
import torch  
from src.environment import LoadBalancerEnv
from src.agents import DQNAgent, RoundRobinAgent

def evaluate(agent, env, episodes=50):
    """
    Tests the agent and returns 3 Critical Metrics:
    1. Average Load: General system health.
    2. Std Deviation: Fairness/Balance across servers.
    3. P99 Latency: The worst-case experience (Tail Latency).
    """
    all_avg_loads = []
    all_std_devs = [] 
    raw_loads = []
    
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 100:
            action = agent.select_action(state)
            next_state, _, done, _, _ = env.step(action)
            
            all_avg_loads.append(np.mean(next_state))    
            all_std_devs.append(np.std(next_state))
            raw_loads.extend(next_state)
            
            state = next_state
            steps += 1
            
    final_avg = np.mean(all_avg_loads)
    final_std = np.mean(all_std_devs)
    final_p99 = np.percentile(raw_loads, 99)
    
    return final_avg, final_std, final_p99

def run_stress_test():
    env = LoadBalancerEnv(num_servers=3)
    
    # Initialize Agents
    dqn_agent = DQNAgent(state_dim=3, action_dim=3)
    rr_agent = RoundRobinAgent(num_servers=3)
    
    print("------------------------------------------------")
    print("STRESS TEST RESULTS")
    print("------------------------------------------------")
    
    model_path = os.path.join("models", "dqn_load_balancer.pth")
    
    if os.path.exists(model_path):
        print(f"Loading pre-trained model: {model_path}")
        # Load weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dqn_agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
        dqn_agent.policy_net.eval() # Set to evaluation mode
        
        # Turn off exploration completely
        dqn_agent.epsilon = 0.0 
        print("Model loaded successfully. Ready to test.")
    else:
        print("Error: Model file not found! Testing with a RANDOM agent.")
        print("   (Please run main.py first to train the agent)")

    print("\nStarting Comprehensive Tests...\n")

    # --- LOW TRAFFIC TEST ---
    env.set_traffic_mode('low')
    dqn_l_avg, dqn_l_std, dqn_l_p99 = evaluate(dqn_agent, env)
    rr_l_avg, rr_l_std, rr_l_p99 = evaluate(rr_agent, env)
    
    print(f"🔹 Low Traffic Results:")
    print(f"   DQN -> Avg: {dqn_l_avg:.3f} | Std: {dqn_l_std:.3f} | P99: {dqn_l_p99:.3f}")
    print(f"   RR  -> Avg: {rr_l_avg:.3f} | Std: {rr_l_std:.3f} | P99: {rr_l_p99:.3f}")

    # --- HIGH TRAFFIC TEST ---
    env.set_traffic_mode('high')
    dqn_h_avg, dqn_h_std, dqn_h_p99 = evaluate(dqn_agent, env)
    rr_h_avg, rr_h_std, rr_h_p99 = evaluate(rr_agent, env)

    print(f"\nHigh Traffic Results:")
    print(f"DQN -> Avg: {dqn_h_avg:.3f} | Std: {dqn_h_std:.3f} | P99: {dqn_h_p99:.3f}")
    print(f"RR  -> Avg: {rr_h_avg:.3f} | Std: {rr_h_std:.3f} | P99: {rr_h_p99:.3f}")

    labels = ['Low Traffic', 'High Traffic']
    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    dqn_avgs = [dqn_l_avg, dqn_h_avg]
    rr_avgs  = [rr_l_avg, rr_h_avg]
    
    dqn_stds = [dqn_l_std, dqn_h_std]
    rr_stds  = [rr_l_std, rr_h_std]
    
    dqn_p99s = [dqn_l_p99, dqn_h_p99]
    rr_p99s  = [rr_l_p99, rr_h_p99]

    # --- CHART 1: AVERAGE LOAD ---
    ax1.bar(x - width/2, dqn_avgs, width, label='DQN', color='royalblue')
    ax1.bar(x + width/2, rr_avgs, width, label='Round Robin', color='orange')
    ax1.set_title('1. General Performance\n(Average Load)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Avg Load')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)

    # --- CHART 2: STANDARD DEVIATION (BALANCE) ---
    ax2.bar(x - width/2, dqn_stds, width, label='DQN', color='royalblue', hatch='//')
    ax2.bar(x + width/2, rr_stds, width, label='Round Robin', color='orange', hatch='//')
    ax2.set_title('2. Fairness / Balance\n(Standard Deviation)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    
    # --- CHART 3: P99 LATENCY ---
    ax3.bar(x - width/2, dqn_p99s, width, label='DQN', color='royalblue', hatch='..')
    ax3.bar(x + width/2, rr_p99s, width, label='Round Robin', color='orange', hatch='..')
    ax3.set_title('3. Worst Case Experience\n(P99 Latency)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('P99 Load')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.legend()
    ax3.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if not os.path.exists("figures"):
        os.makedirs("figures")
        print("'figures' folder created.")

    save_path = os.path.join("figures", "stress_test_results.png")
    plt.savefig(save_path)
    print(f"\nComprehensive Chart saved: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_stress_test()