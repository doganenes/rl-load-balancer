import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class ReplayBuffer:
    """Stores and samples past transitions for training stability."""
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

class DQN(nn.Module):
    """Standard DQN architecture."""
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

class DuelingDQN(nn.Module):
    """Dueling DQN with separate value and advantage streams."""
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
    """
    DQN-based agent supporting Standard and Dueling architectures,
    with optional target network and replay memory.
    """    
    def __init__(self, state_dim, action_dim, lr=0.0001, gamma=0.95, 
                 use_dueling=True, use_target_network=True, use_replay_memory=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_dueling = use_dueling
        self.use_target_network = use_target_network
        self.use_replay_memory = use_replay_memory
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        config = []
        if self.use_dueling: config.append("Dueling")
        else: config.append("Standard")
        if not self.use_target_network: config.append("NoTarget")
        if not self.use_replay_memory: config.append("NoReplay")
        print(f"Agent Initialized: {'-'.join(config)}")
        
        if self.use_dueling:
            self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
            self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        else:
            self.policy_net = DQN(state_dim, action_dim).to(self.device)
            self.target_net = DQN(state_dim, action_dim).to(self.device)
        
        if self.use_target_network:
            self.update_target_network()
            self.target_net.eval()
        else:
            self.target_net = self.policy_net
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        if self.use_replay_memory:
            self.memory = ReplayBuffer(50000)
        else:
            self.memory = None
            self.last_transition = None
        
        self.batch_size = 128
        self.gamma = gamma
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_arr = np.array(state).flatten()
                state_t = torch.FloatTensor(state_arr).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                return q_values.argmax().item()

    def learn(self):
        """Updates network parameters using sampled experiences."""
        if not self.use_replay_memory or (self.use_replay_memory and len(self.memory) < self.batch_size):
            return
        
        if self.use_replay_memory:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        else:
            if self.last_transition is None:
                return
            states, actions, rewards, next_states, dones = [self.last_transition[0]], [self.last_transition[1]], [self.last_transition[2]], [self.last_transition[3]], [self.last_transition[4]]
        
        states = np.stack(states)
        next_states = np.stack(next_states)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            if self.use_target_network:
                next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            else:
                next_q_values = self.policy_net(next_states).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        if self.use_target_network:
            self.update_target_network()

    def update_target_network(self):
        """Soft update of target network"""
        tau = 0.005
        target_state_dict = self.target_net.state_dict()
        policy_state_dict = self.policy_net.state_dict()
        for key in policy_state_dict:
            target_state_dict[key] = policy_state_dict[key] * tau + target_state_dict[key] * (1 - tau)
        self.target_net.load_state_dict(target_state_dict)
    
    def update_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Stores transition in replay memory or keeps last transition."""
        if self.use_replay_memory:
            self.memory.push(state, action, reward, next_state, done)
        else:
            state = np.array(state).flatten()
            next_state = np.array(next_state).flatten()
            self.last_transition = (state, action, reward, next_state, done)

class RoundRobinAgent:
    """Round Robin Agent for Load Balancing."""
    def __init__(self, num_servers=3):
        self.num_servers = num_servers
        self.current_index = 0

    def select_action(self, state):
        action = self.current_index
        self.current_index = (self.current_index + 1) % self.num_servers
        return action


class LeastConnectionsAgent:
    """Baseline selecting the least loaded server."""
    def __init__(self):
        pass

    def select_action(self, state):
        return np.argmin(state)