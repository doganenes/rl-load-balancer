<img width="810" height="195" alt="Image" src="https://github.com/user-attachments/assets/bb9e353e-661e-4c41-9e2a-bac9bc4a22ea" />

# RL-Based Dynamic Load Balancing in Distributed Systems

This project implements an autonomous Reinforcement Learning (RL) framework designed to optimize workload distribution across a three-server cluster. By modeling the load balancing problem as a Markov Decision Process (MDP), the system learns an adaptive policy that responds to real-time traffic fluctuations and resource saturation.

## System Methodology
The core of the project is the interaction between a central RL Agent and a simulated cluster environment developed using the `Gymnasium` library.
### RL Architectures
The project evaluates two primary neural network-based RL architectures:

- **Standard DQN**  
  Approximates the Q-value function to handle the continuous state space of server loads.

- **Dueling DQN**  
  Decouples the State Value $V(s)$ from the Action Advantage $A(s,a)$, allowing the agent to identify high-risk states regardless of the specific routing decision.
<img width="797" height="323" alt="Image" src="https://github.com/user-attachments/assets/0a530344-b279-40b7-ae0f-5f620b7c9cfe" />

## Architecture Performance Comparison
The Standard DQN and Dueling DQN were compared head-to-head under identical conditions following hyperparameter optimization to verify architectural superiority.
### 1. Steady-State Comparison (Low Traffic)
<img width="598" height="401" alt="Image" src="https://github.com/user-attachments/assets/939d1411-d060-43a7-ab09-925a5a97ed8b" />
Analysis: When system demand matches processing capacity, both agents converge to a stable operating regime with nearly identical performance.

### 2. Over-Saturation Comparison (High Traffic)
<img width="601" height="402" alt="Image" src="https://github.com/user-attachments/assets/27edd678-b494-48af-af0d-c48762661d4c" />
Analysis: The Standard DQN exhibits noticeable instability due to overestimation bias. In contrast, the Dueling DQN maintains a significantly more stable and robust response despite persistent overload.


## Experimental Results
### 1. Hyperparameter Optimization
A parallelized grid search was conducted using multiprocessing to identify the most stable RL parameters.  
The results identified $\alpha = 0.001$ and $\gamma = 0.99$ as the optimal configuration for high-traffic stability.
| Rank | Learning Rate | Gamma | Architecture  | Average Reward |
|------|---------------|-------|---------------|----------------|
| 1    | 0.001         | 0.99  | Dueling DQN   | -62.98         |
| 2    | 0.001         | 0.95  | Dueling DQN   | -66.02         |
| 3    | 0.0005        | 0.99  | Dueling DQN   | -70.18         |
| 4    | 0.001         | 0.99  | Standard DQN  | -70.27         |
| 5    | 0.001         | 0.90  | Dueling DQN   | -73.71         |
| 6    | 0.0005        | 0.95  | Standard DQN  | -74.21         |
| 7    | 0.0001        | 0.99  | Standard DQN  | -74.31         |
| 8    | 0.0005        | 0.90  | Standard DQN  | -75.92         |
| 9    | 0.001         | 0.90  | Standard DQN  | -76.34         |
| 10   | 0.0001        | 0.99  | Dueling DQN   | -76.60         |
| 11   | 0.0005        | 0.90  | Dueling DQN   | -77.11         |
| 12   | 0.0001        | 0.95  | Standard DQN  | -78.42         |
| 13   | 0.0005        | 0.95  | Dueling DQN   | -78.54         |
| 14   | 0.0001        | 0.90  | Dueling DQN   | -78.60         |
| 15   | 0.0005        | 0.99  | Standard DQN  | -79.78         |
| 16   | 0.0001        | 0.90  | Standard DQN  | -81.26         |
| 17   | 0.001         | 0.95  | Standard DQN  | -82.22         |
| 18   | 0.0001        | 0.95  | Dueling DQN   | -86.12         |

### 2. Ablation Study: Stability Analysis
We evaluated the necessity of RL stabilization mechanisms—specifically **Experience Replay** and **Target Networks with soft updates** ($\tau = 0.005$).  
The ablation study was conducted across two traffic regimes to isolate the impact of each component.

#### Low Traffic Stability
<img width="1000" height="600" alt="Image" src="https://github.com/user-attachments/assets/36959f6c-3cde-4b0b-b7bb-0f58084725ee" />
Under low-load conditions, all architectures except **No Replay Memory** exhibited relatively stable convergence.  
This suggests that when system resources are abundant, the decoupling advantage of the Dueling architecture is less pronounced.  


#### High Traffic Robustness
The true value of the **Dueling DQN** and stabilization components was unveiled during high-stress scenarios (116% saturation).  
Without **Experience Replay Memory** (orange line), the agent failed to break temporal correlations, leading to complete training stagnation.  
<img width="1000" height="600" alt="Image" src="https://github.com/user-attachments/assets/38992bab-0a8d-42d8-9aa9-6a0854fbe558" />

### 3. Stress Test Evaluation
The trained RL policy was compared against industry-standard heuristics: **Least Connections** and **Round Robin**.
<img width="1800" height="600" alt="Image" src="https://github.com/user-attachments/assets/a72839b3-1295-41a6-a7eb-460588cc61a6" />
Under high traffic, the RL agent maintains superior fairness (0.237 Std Dev) and minimizes P99 latency compared to static baselines.

### 4. Real-World Validation
To ensure the simulation's realism, the server load vectors generated by the RL agent were compared against **Mendeley Data workload traces**.
| Metric             | Similarity (%) |
|-------------------|----------------|
| Mean (Average)     | 90.21          |
| Standard Deviation | 5.42           |
| Min. Similarity    | 76.60          |
| Max. Similarity    | 98.45          |
Result: The agent's learned policy achieved a 90.21% mean similarity with real-world server states.

## Project Structure
- **src/environment.py**  
  Custom Gymnasium environment for the 3-server cluster.
- **src/agents.py**  
  Implementation of Standard and Dueling DQN RL agents.
- **tune2.py**  
  Multiprocessing script for parallel hyperparameter search.
- **visualize2.py**  
  Simulation visualizer producing real-time load distribution GIFs.
- **benchmark.py**  
  Script for calculating similarity metrics against Mendeley Data.
