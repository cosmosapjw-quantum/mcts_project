# test_callback.py
import mcts_py

def simple_callback(batch_input):
    print("Callback received:", batch_input)
    return [([1.0/225] * 225, 0.0)]

def main():
    print("Creating config...")
    cfg = mcts_py.MCTSConfig()
    cfg.num_simulations = 1
    cfg.c_puct = 1.0
    cfg.num_threads = 1  # Single thread to simplify debugging
    
    print("Creating wrapper...")
    wrapper = mcts_py.MCTSWrapper(cfg, boardSize=15, use_omok=False)
    
    print("Setting callback...")
    wrapper.set_infer_function(simple_callback)
    
    print("Running search...")
    wrapper.run_search()
    
    print("Done")

if __name__ == "__main__":
    main()