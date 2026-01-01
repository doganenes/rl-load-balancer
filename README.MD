# RL-Based Dynamic Load Balancing in Distributed Systems
![Tech Stack](https://github.com/user-attachments/assets/2bd64c66-0398-46c3-bf03-f2b9d5d0438d)


This project implements an autonomous Reinforcement Learning (RL) framework designed to optimize workload distribution across a three-server cluster. By modeling the load balancing problem as a **Markov Decision Process (MDP)**, the system learns an adaptive policy that responds to real-time traffic fluctuations and resource saturation.

---

## System Methodology

The core of the project is the interaction between a central RL Agent and a simulated cluster environment developed using the `Gymnasium` library.

### Architecture

![System Architecture](https://github.com/user-attachments/assets/0a530344-b279-40b7-ae0f-5f620b7c9cfe)

The project evaluates two primary neural network-based RL architectures:

- **Standard DQN**  
  Approximates the Q-value function to handle the continuous state space of server loads.

- **Dueling DQN**  
  Decouples the State Value $V(s)$ from the Action Advantage $A(s,a)$, allowing the agent to identify high-risk states regardless of the specific routing decision.

---

## Architecture Performance Comparison

The Standard DQN and Dueling DQN were compared head-to-head under identical conditions following hyperparameter optimization to verify architectural superiority.

### 1. Steady-State Comparison (Low Traffic)
![Architecture Comparision under Low Traffic](https://github.com/user-attachments/assets/db6811f7-5d96-4211-bbd7-a70fcf0a9c66)


**Analysis:** When system demand matches processing capacity, both agents converge to a stable operating regime with nearly identical performance.

### 2. Over-Saturation Comparison (High Traffic)
![Architecture Comparision under High Traffic](https://github.com/user-attachments/assets/8392ef79-ec5c-4d51-ac47-8dafe39cbb38)

**Analysis:** The Standard DQN exhibits noticeable instability due to overestimation bias. In contrast, the Dueling DQN maintains a significantly more stable and robust response despite persistent overload.

---

## Project Structure

- **src/environment.py**  
  A custom Gymnasium environment that simulates a 3-server cluster, managing state transitions based on server processing rates and traffic modes (Low/High).

- **src/agents.py**  
  Implementation of the Reinforcement Learning agents, including the Standard DQN and Dueling DQN neural network architectures, as well as baseline heuristics such as Round Robin and Least Connections.

- **main.py**  
  The primary script for training the Dueling DQN agent, handling the training loop, model saving, and generating reward history plots.

- **tune.py**  
  A high-performance multiprocessing script used to parallelize a grid search over learning rates and discount factors to identify optimal hyperparameters.

- **compare.py**  
  A specialized script for performing head-to-head performance comparisons between Standard and Dueling architectures under identical high-traffic conditions.

- **ablation.py**  
  A diagnostic script that performs an ablation study by systematically disabling core components like the Target Network or Replay Memory to quantify their impact on training stability.

- **test.py**  
  A comprehensive stress test script that evaluates trained agents against traditional baselines using metrics like average load, load standard deviation (fairness), and P99 load.

- **visualize.py**  
  A simulation utility that produces real-time load distribution GIFs and step-by-step visualizations of server CPU utilization.

- **benchmark.py**  
  A validation tool that calculates Euclidean distance and similarity percentages to compare simulation telemetry against Mendeley Data industrial benchmark traces.

---

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

---

### 2. Stress Test Evaluation

The trained RL policy was compared against industry-standard heuristics: **Least Connections** and **Round Robin**.

![Stress Test](https://github.com/user-attachments/assets/a72839b3-1295-41a6-a7eb-460588cc61a6)

**Analysis:** Under high traffic, the RL agent maintains superior fairness (0.237 Std Dev) and minimizes P99 latency compared to static baselines.

---

### 3. Real-World Validation

To ensure the simulation's realism, the server load vectors generated by the RL agent were compared against **Mendeley Data workload traces**.

| Metric             | Similarity (%) |
|-------------------|----------------|
| Mean (Average)     | 90.21          |
| Standard Deviation | 5.42           |
| Min. Similarity    | 76.60          |
| Max. Similarity    | 98.45          |

**Result:** The agent's learned policy achieved a 90.21% mean similarity with real-world server states.

---

### 4. Real-Time Load Visualization

The following visualizations illustrate the agent's routing behavior at the system level during the testing phase.  
These were generated using **ImageIO** to capture real-time load distributions.

![Low Traffic Routing](https://github.com/user-attachments/assets/2c014aa8-9c79-4777-a699-baf8c98a8035)  
*Figure: RL agent routing behavior under low traffic.*

![High Traffic Routing](https://github.com/user-attachments/assets/e10959b7-e343-4c3d-98cd-43bce6509be6)  
*Figure: RL agent routing behavior under high traffic.*
