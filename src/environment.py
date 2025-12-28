import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LoadBalancerEnv(gym.Env):
    def __init__(self, num_servers=3, max_steps=100):
        super().__init__()
        self.num_servers = num_servers
        self.max_steps = max_steps
        self.current_step = 0

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(num_servers,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(num_servers)

        self.processing_rate = np.array([0.10, 0.10, 0.10])
        
        self.state = np.zeros(num_servers)
        self.set_traffic_mode("low")

    def set_traffic_mode(self, mode):
        if mode == "low":
            self.min_load = 0.25
            self.max_load = 0.35

        elif mode == "high":
            self.min_load = 0.30
            self.max_load = 0.40
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.state = np.zeros(self.num_servers)
        self.current_step = 0
        
        return self.state.astype(np.float32), {}


    def step(self, action):
        self.current_step += 1

        incoming = np.random.uniform(self.min_load, self.max_load)
        self.state[action] += incoming

        
        std_dev = np.std(self.state)
        base_reward = - (std_dev * 2.0)
        
        max_current_load = np.max(self.state)
        risk_penalty = 0.0
        
        if max_current_load > 0.8:
            risk_penalty = (max_current_load - 0.8) * 50.0

        # Crash Kontrolü
        terminated = False
        
        if max_current_load > 1.0:
            total_reward = -50.0 
            terminated = True
            self.state[action] = 1.0
        else:
            total_reward = base_reward - risk_penalty

       
        self.state = self.state - self.processing_rate
        self.state = np.maximum(0.0, self.state)

        if self.current_step >= self.max_steps:
            truncated = True
        else:
            truncated = False

        return self.state.astype(np.float32), total_reward, terminated, truncated, {}

    def render(self):
        print(f"Step: {self.current_step} | Loads: {self.state}")