# model.py update
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedGomokuNet(nn.Module):
    def __init__(self, board_size=15, policy_dim=225, num_history_moves=3):
        super().__init__()
        self.board_size = board_size
        self.num_history_moves = num_history_moves
        
        # New input dimensions: 
        # board_size*board_size (current board)
        # + 1 (player flag)
        # + 2*num_history_moves (previous N moves for both players)
        # + 2 (attack/defense scores)
        input_dim = board_size*board_size + 1 + 2*num_history_moves + 2
        
        hidden_dim = 256
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Output heads:
        self.policy_head = nn.Linear(hidden_dim, policy_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x shape: [batch, input_dim]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # policy
        policy_logits = self.policy_head(x)  # shape [batch, policy_dim]
        # value
        value = torch.tanh(self.value_head(x))  # shape [batch, 1]
        return policy_logits, value