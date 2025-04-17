#\!/usr/bin/env python3
# Ultra minimal test for just testing startup and shutdown

import sys
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import mcts_py
import time

print("Creating minimal model")
# Define a minimal model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(235, 32)
        self.policy = nn.Linear(32, 225)
        self.value = nn.Linear(32, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        policy = self.policy(x)
        value = torch.tanh(self.value(x))
        return policy, value

model = SimpleModel()
model.eval()

print("Setting up signal handlers")
mcts_py.register_signal_handlers()

print("Creating a very minimal MCTS config")
cfg = mcts_py.MCTSConfig()
cfg.num_simulations = 100  # Very small number
cfg.num_threads = 1        # Single thread
cfg.parallel_leaf_batch_size = 1  # No batching

print("Creating wrapper")
wrapper = None
try:
    wrapper = mcts_py.MCTSWrapper(cfg, 15, False, False, 0, False)
    
    print("Setting model")
    wrapper.set_infer_function(model)
    
    print("All setup complete - now explicitly closing")
    del wrapper
    wrapper = None
    
    # Force garbage collection
    gc.collect()
    
    print("Test succeeded")
except Exception as e:
    print(f"Test FAILED with error: {e}")
    
print("Exiting cleanly")
