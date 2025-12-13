import matplotlib.pyplot as plt
import numpy as np

# Standard single training plot (Required for Main.py).
def plot_learning_curve(rewards, filename='training_curve.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='DQN Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Training Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_learning_curve_with_shadow(all_rewards, filename='academic_training_curve.png'):    
    data = np.array(all_rewards)     
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    x = np.arange(len(mean))
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(x, mean, color='b', label='DQN (Mean Reward)')
    plt.fill_between(x, mean - std, mean + std, color='b', alpha=0.2, label='Standard Deviation')
    
    plt.title(f'Training Performance (Averaged over {len(data)} runs)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(filename)
    plt.close()
    print(f"Chart saved: {filename}")

#DQN vs Round Robin comparison bar chart.
def plot_comparison(dqn_metrics, rr_metrics, filename='comparison_result.png'):
    labels = ['Avg Response Time', 'Throughput']
    dqn_vals = [dqn_metrics['avg_response'], dqn_metrics['throughput']]
    rr_vals = [rr_metrics['avg_response'], rr_metrics['throughput']]
    
    x = range(len(labels))
    width = 0.35
    
    plt.figure(figsize=(8, 5))
    plt.bar([i - width/2 for i in x], dqn_vals, width, label='DQN')
    plt.bar([i + width/2 for i in x], rr_vals, width, label='Round Robin')
    
    plt.xticks(x, labels)
    plt.legend()
    plt.title('DQN vs Round Robin Comparison')
    plt.savefig(filename)
    plt.close()