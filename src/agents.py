import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# --- REPLAY BUFFER ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state = np.array(state).flatten()
        next_state = np.array(next_state).flatten()
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# STANDARD DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.01),  
            nn.Linear(64, 64),
            nn.LeakyReLU(0.01),  
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        return self.fc(x)

# DUELING DQN
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.01)
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.0001, gamma=0.95, use_dueling=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_dueling = use_dueling
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Agent Initialized! Model: {'Dueling DQN' if use_dueling else 'Standard DQN'}")
        
        if self.use_dueling:
            self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
            self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        else:
            self.policy_net = DQN(state_dim, action_dim).to(self.device)
            self.target_net = DQN(state_dim, action_dim).to(self.device)
            
        self.update_target_network()
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(50000)
        self.batch_size = 128
        self.gamma = gamma
        
        # Epsilon Settings
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.985

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_arr = np.array(state).flatten()
                state_t = torch.FloatTensor(state_arr).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                return q_values.argmax().item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = np.stack(states)
        next_states = np.stack(next_states)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.update_target_network()

    def update_target_network(self):
        tau = 0.005
        target_state_dict = self.target_net.state_dict()
        policy_state_dict = self.policy_net.state_dict()
        for key in policy_state_dict:
            target_state_dict[key] = policy_state_dict[key] * tau + target_state_dict[key] * (1 - tau)
        self.target_net.load_state_dict(target_state_dict)
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class RoundRobinAgent:
    def __init__(self, num_servers=3):
        self.num_servers = num_servers
        self.current_index = 0

    def select_action(self, state):
        action = self.current_index
        self.current_index = (self.current_index + 1) % self.num_servers
        return action


class LeastConnectionsAgent:
    def __init__(self):
        pass

    def select_action(self, state):
        return np.argmin(state)