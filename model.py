import torch
import torch.nn as nn
import numpy as np

# Calculate the range of values for uniform distributions
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class VQNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self,  
                 action_size,
                 state_size,
                 frames_num,
                 channels = [128, 256, 256],
                 num_layers = [1024, 512]
                 ):
        """Initialize parameters and build model.
        
        Params
        ======
            action_size (int): Dimension of each action
            state_size (tuple): Shape of each state  
            frames_num (int): Number of stacked states (images) 
            channels (list): Number of channels for each convolution layer
            num_layers (list): Number of nodes for each linear layer  
        """
        # Calls the init function of nn.Module
        super(VQNetwork, self).__init__()
        # Container with convolutional layers 
        self.conv_layer = nn.Sequential(
            nn.Conv3d(3, channels[0], kernel_size=(1, 3, 3), stride=(1, 3, 3)),
            nn.BatchNorm3d(channels[0]), # batch norm layer
            nn.ReLU(),
            nn.Conv3d(channels[0], channels[1], kernel_size=(1, 3, 3), stride=(1, 3, 3)),
            nn.BatchNorm3d(channels[1]),
            nn.ReLU(),
            nn.Conv3d(channels[1], channels[2], kernel_size=(1, 3, 3), stride=(1, 3, 3)),
            nn.BatchNorm3d(channels[2]),
            nn.ReLU())
        
        # Container with linear layers 
        self.fc_layer = nn.Sequential(
            nn.Linear(self.conv_size_out(state_size, frames_num), num_layers[0]),
            nn.BatchNorm1d(num_layers[0]), # batch norm layer 
            nn.Dropout(p=0.2), # dropout layer
            nn.ReLU(), 
            nn.Linear(num_layers[0], num_layers[1]),
            nn.BatchNorm1d(num_layers[1]),
            nn.Dropout(p=0.1),
            nn.ReLU())
                  
        # Get state-value
        self.value_func_layer = nn.Linear(num_layers[1], 1)
        # Get advantages for each action
        self.adv_func_layer = nn.Linear(num_layers[1], action_size)
        self.reset_parameters()
     
    def reset_parameters(self):
        # Apply to layers the specified weight initialization
        self.conv_layer[0].weight.data.uniform_(*hidden_init(self.conv_layer[0]))
        self.conv_layer[3].weight.data.uniform_(*hidden_init(self.conv_layer[3]))
        self.conv_layer[6].weight.data.uniform_(*hidden_init(self.conv_layer[6]))
        self.fc_layer[0].weight.data.uniform_(*hidden_init(self.fc_layer[0]))
        self.fc_layer[4].weight.data.uniform_(*hidden_init(self.fc_layer[4]))
        self.value_func_layer.weight.data.uniform_(-3e-3, 3e-3)
        self.adv_func_layer.weight.data.uniform_(-3e-3, 3e-3)
    
    def conv_size_out(self, state_size, frames_num):
        """Calculate the number of parameters for the first linear layer."""
        output = list(state_size)
        del output[3]
        output.insert(1, state_size[3])
        output.insert(2, frames_num)
        inp = torch.randn(output)
        output = self.conv_layer(inp)
        output = np.array((output.shape[1:]))
        output = np.prod(output)
        return output
        
    def forward(self, observation):
        """Build a network that maps state -> action values."""
        x = self.conv_layer(observation)
        # Prep for linear layer
        # Flatten the inputs into a vector
        x = x.view(x.shape[0], -1)
        x = self.fc_layer(x)
        v_s = self.value_func_layer(x)
        a_sa = self.adv_func_layer(x)
        return v_s + (a_sa - a_sa.mean())    