# example_train_script.py
import torch
from torch import optim
from model import SimpleGomokuNet

def train_on_data(data_buffer, net, epochs=1):
    """
    data_buffer: list of (board_input, move, winner, etc.)
    net: your neural net
    This is a trivial placeholder example.
    """
    opt = optim.Adam(net.parameters(), lr=1e-3)

    for ep in range(epochs):
        total_loss = 0.0
        for (boardInput, move, winner) in data_buffer:
            # Convert to tensor, do forward pass, compute loss...
            pass
        print("Epoch", ep, "loss=", total_loss)

def main():
    # Suppose after self_play we have global_data_buffer:
    # You can import the buffer or pass it in
    from self_play import global_data_buffer
    
    net = SimpleGomokuNet(board_size=15, policy_dim=225)
    train_on_data(global_data_buffer, net, epochs=3)
    # Save
    torch.save(net.state_dict(), "my_model.pth")

if __name__=="__main__":
    main()
