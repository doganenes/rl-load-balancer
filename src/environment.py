import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LoadBalancerEnv(gym.Env):
    def __init__(self, num_servers=3):
        super(LoadBalancerEnv, self).__init__()
        self.num_servers = num_servers
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_servers,), dtype=np.float32)
        
        self.action_space = spaces.Discrete(num_servers)
        
        self.state = np.zeros(num_servers)
        
        self.arrival_rate = 0.25

    def set_traffic_mode(self, mode):
        """
        Modifies traffic intensity for test scenarios.
        Note: Default Processing Power is 0.05.
        """
        if mode == 'low':
            self.arrival_rate = 0.07  
        elif mode == 'high':
            self.arrival_rate = 0.35  
        else:
            self.arrival_rate = 0.15

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.state = np.zeros(self.num_servers)
        return self.state, {}

    def step(self, action):
        self.state[action] = min(1.0, self.state[action] + self.arrival_rate)  
        processing_power = 0.05
        self.state = np.maximum(0, self.state - processing_power)
        
        avg_load = np.mean(self.state)
        std_dev = np.std(self.state)
        max_load = np.max(self.state)
        
        reward = 1.0 - (avg_load + std_dev + 0.5 * max_load)
        
        terminated = False
        truncated = False
        
        return self.state, reward, terminated, truncated, {}