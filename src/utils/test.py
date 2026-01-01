import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import logging

from src.environment import LoadBalancerEnv
from src.agents import DQNAgent, RoundRobinAgent, LeastConnectionsAgent

if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename="logs/stress_test_results.log",
    filemode="w",
    level=logging.INFO,
    format="%(message)s"
)

def evaluate(agent, env, episodes=50, agent_name="Agent", traffic_mode="low"):
    """
     Tests the agent and returns 3 critical metrics:
    - Average Load
    - Standard Deviation of Load
    - 99th percentile (P99) of Load
    """
    all_avg_loads = []
    all_std_devs = []
    raw_loads = []


    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < 100:
            action = agent.select_action(state)
            next_state, _, done, _, _ = env.step(action)

            avg_load = np.mean(next_state)
            std_load = np.std(next_state)

            all_avg_loads.append(avg_load)
            all_std_devs.append(std_load)
            raw_loads.extend(next_state)

            logging.info(
            f"[{traffic_mode.upper()}][{agent_name}][Episode {ep+1}][Step {steps+1}] "
            f"Loads={np.round(next_state,3)} Avg={avg_load:.3f} Std={std_load:.3f}"
            )

            state = next_state
            steps += 1

    final_avg = np.mean(all_avg_loads)
    final_std = np.mean(all_std_devs)
    final_p99 = np.percentile(raw_loads, 99)

    return final_avg, final_std, final_p99

def run_stress_test():
    env = LoadBalancerEnv(num_servers=3)
    state_dim = env.observation_space.shape[0]
    dqn_agent = DQNAgent(state_dim=state_dim, action_dim=3, use_dueling=True)
    lc_agent = LeastConnectionsAgent()
    rr_agent = RoundRobinAgent(num_servers=3)

    model_path = "models/dqn_load_balancer_11.pth"

    if os.path.exists(model_path):
        print(f"Loading pre-trained model: {model_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dqn_agent.policy_net.load_state_dict(
            torch.load(model_path, map_location=device)
        )

        dqn_agent.policy_net.eval()
        dqn_agent.epsilon = 0.0
        print("Model loaded successfully. Ready to test.")
    else:
        print("Error: Model file not found!")
        return

    print("\nStarting Comprehensive Tests...\n")

    env.set_traffic_mode("low")
    print("🔹 Testing Low Traffic...")

    dqn_l_avg, dqn_l_std, dqn_l_p99 = evaluate(
        dqn_agent, env, agent_name="DQN", traffic_mode="low"
    )
    lc_l_avg, lc_l_std, lc_l_p99 = evaluate(
        lc_agent, env, agent_name="LeastConn", traffic_mode="low"
    )
    rr_l_avg, rr_l_std, rr_l_p99 = evaluate(
        rr_agent, env, agent_name="RoundRobin", traffic_mode="low"
    )

    print(f"   DQN        -> Avg: {dqn_l_avg:.3f} | Std: {dqn_l_std:.3f}")
    print(f"   Least Conn -> Avg: {lc_l_avg:.3f} | Std: {lc_l_std:.3f}")
    print(f"   RR         -> Avg: {rr_l_avg:.3f} | Std: {rr_l_std:.3f}")

    env.set_traffic_mode("high")
    print("\nTesting High Traffic...")

    dqn_h_avg, dqn_h_std, dqn_h_p99 = evaluate(
        dqn_agent, env, agent_name="DQN", traffic_mode="high"
    )
    lc_h_avg, lc_h_std, lc_h_p99 = evaluate(
        lc_agent, env, agent_name="LeastConn", traffic_mode="high"
    )
    rr_h_avg, rr_h_std, rr_h_p99 = evaluate(
        rr_agent, env, agent_name="RoundRobin", traffic_mode="high"
    )

    print(f"   DQN        -> Avg: {dqn_h_avg:.3f} | Std: {dqn_h_std:.3f}")
    print(f"   Least Conn -> Avg: {lc_h_avg:.3f} | Std: {lc_h_std:.3f}")
    print(f"   RR         -> Avg: {rr_h_avg:.3f} | Std: {rr_h_std:.3f}")

    labels = ["Low Traffic", "High Traffic"]
    x = np.arange(len(labels))
    width = 0.25

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    ax1.bar(x - width, [dqn_l_avg, dqn_h_avg], width, label="DQN")
    ax1.bar(x, [lc_l_avg, lc_h_avg], width, label="Least Conn")
    ax1.bar(x + width, [rr_l_avg, rr_h_avg], width, label="Round Robin")
    ax1.set_title("Average Load")
    ax1.legend()

    ax2.bar(x - width, [dqn_l_std, dqn_h_std], width)
    ax2.bar(x, [lc_l_std, lc_h_std], width)
    ax2.bar(x + width, [rr_l_std, rr_h_std], width)
    ax2.set_title("Std Deviation")

    ax3.bar(x - width, [dqn_l_p99, dqn_h_p99], width)
    ax3.bar(x, [lc_l_p99, lc_h_p99], width)
    ax3.bar(x + width, [rr_l_p99, rr_h_p99], width)
    ax3.set_title("P99 Load")

    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

    plt.tight_layout()

    if not os.path.exists("figures"):
        os.makedirs("figures")

    plt.savefig("figures/stress_test_results.png")
    plt.show()

    print("\nLogs written to: logs/stress_test_results.log")

if __name__ == "__main__":
    run_stress_test()
