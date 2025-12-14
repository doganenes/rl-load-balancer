import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LoadBalancerEnv(gym.Env):
    """
    Realistic Load Balancer Simulation Environment
    - Allows overload (Load > 1.0) to simulate crashes/failures.
    - Risk-aware reward function.
    - Gymnasium compliant (terminated/truncated logic).
    """
    def __init__(self, num_servers=3):
        super(LoadBalancerEnv, self).__init__()
        self.num_servers = num_servers
        self.observation_space = spaces.Box(low=0, high=10, shape=(num_servers,), dtype=np.float32)
        self.action_space = spaces.Discrete(num_servers) 
        self.state = np.zeros(num_servers)
        
        self.processing_rate = 0.1
        self.min_req_size = 0.05
        self.max_req_size = 0.15

    def set_traffic_mode(self, mode):
        """
        Modifies traffic intensity for test scenarios.
        """
        if mode == 'low':
            self.min_req_size = 0.05
            self.max_req_size = 0.15
            self.processing_rate = 0.08
            
        elif mode == 'high':
            self.min_req_size = 0.20  
            self.max_req_size = 0.40  
            self.processing_rate = 0.10
            
        else:
            self.min_req_size = 0.10
            self.max_req_size = 0.20
            self.processing_rate = 0.1

    def reset(self, seed=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.state = np.zeros(self.num_servers)
        return self.state, {}

    def step(self, action):
        current_request_size = np.random.uniform(self.min_req_size, self.max_req_size)
        
        self.state[action] += current_request_size

        avg_load = np.mean(self.state)       
        std_dev = np.std(self.state)         
        max_load = np.max(self.state)        

        w_perf = 1.0   
        w_fair = 2.5   
        w_risk = 2.0   

        reward = - (w_perf * avg_load + w_fair * std_dev + w_risk * max_load)

        terminated = False
        if max_load > 1.0:
            reward -= 50.0
            terminated = True
        
        self.state = np.maximum(0, self.state - self.processing_rate)
        
        truncated = False
        
        return self.state, reward, terminated, truncated, {}