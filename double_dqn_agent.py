import numpy as np
import random
from collections import namedtuple, deque

from model_vb import VQNetwork

import torch
import torchvision.transforms as T
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e4)    # replay buffer size
BATCH_SIZE = 64           # minibatch size
GAMMA = 0.99              # discount factor
TAU = 1e-3                # for soft update of target parameters
LR = 5e-4                 # learning rate
LR_DECAY = 0.99999        # multiplicative factor of learning rate decay
UPDATE_EVERY = 4          # how often to update the network

# Device to run the training on. Must be cuda' or 'cpu'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, frames_num):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            frames_num (int): number of stacked RGB images
        """
        self.state_size = state_size
        self.action_size = action_size
        self.frames_num = frames_num
        
        # Q-Network
        self.qnetwork_local = VQNetwork(action_size, state_size, frames_num).to(device)
        self.qnetwork_target = VQNetwork(action_size, state_size, frames_num).to(device)
        
        # Optimization method
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        # Learning rate schedule
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[570], gamma=0.02)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()                           
                self.learn(experiences, GAMMA)
    
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # Convert a numpy array to a new float tensor of shape (1, state_size) and upload it to device
        state = torch.from_numpy(state).float().to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_selection = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
        self.qnetwork_local.train()
        
        # Get predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, action_selection)
                
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        # Get values for corresponding actions along the rows action-value matrix output: 
            # (BATCH_SIZE, action_size) -> (BATCH_SIZE, 1)
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.smooth_l1_loss(Q_expected, Q_targets)

        # Minimize the loss
        # Clear the gradients, do this because gradients are accumulated
        self.optimizer.zero_grad()
        # Perfom a backward pass through the network to calculate the gradients (backpropagate the error)
        loss.backward()
        
        # Take a step with optimaizer to update the weights
        self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory.
        
        Returns
        ====== 
            batch of experiences (tuple): 
                states (torch.float): 2-D tensor of shape (batch_size, state_size)
                actions (torch.long): 2-D tensor of shape (batch_size, 1)
                rewards (torch.float): 2-D tensor of shape (batch_size, 1)
                next_states (torch.float): 2-D tensor of shape (batch_size, 1)
                dones (torch.float): 2-D tensor of shape (batch_size, 1)
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)