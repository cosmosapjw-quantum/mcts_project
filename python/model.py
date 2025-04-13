# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGomokuNet(nn.Module):
    def __init__(self, board_size=15, policy_dim=225):
        super().__init__()
        self.board_size = board_size
        # We'll pretend each board is flattened: board_size*board_size 
        # plus 3 extra floats (chosenMove, attack, defense) => total input size
        input_dim = board_size*board_size + 3
        
        hidden_dim = 256
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Output heads:
        self.policy_head = nn.Linear(hidden_dim, policy_dim)
        self.value_head  = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x shape: [batch, input_dim]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # policy
        policy_logits = self.policy_head(x)  # shape [batch, policy_dim]
        # value
        value = torch.tanh(self.value_head(x))  # shape [batch, 1]
        return policy_logits, value
