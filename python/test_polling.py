# test_polling.py
import mcts_py
import time

def process_nn_requests(nn_interface):
    if nn_interface.has_pending_requests():
        requests = nn_interface.get_pending_requests()
        print(f"Processing {len(requests)} NN requests")
        
        for i, req in enumerate(requests):
            # Simple policy and value for testing
            policy = [1.0/225] * 225
            value = 0.0
            nn_interface.submit_result(i, policy, value)
        
        return True
    return False

def main():
    print("Creating config...")
    cfg = mcts_py.MCTSConfig()
    cfg.num_simulations = 5
    cfg.c_puct = 1.0
    cfg.num_threads = 1
    
    print("Creating wrapper...")
    wrapper = mcts_py.MCTSWrapper(cfg, boardSize=15, use_omok=False)
    
    # Get access to the NN interface
    nn_interface = wrapper._get_nn_interface()
    
    # Start a background thread for search
    import threading
    def run_search():
        print("Running search...")
        wrapper.run_search()
        print("Search completed")
    
    thread = threading.Thread(target=run_search)
    thread.start()
    
    # Process NN requests in the main Python thread
    while thread.is_alive():
        if process_nn_requests(nn_interface):
            print("Processed batch")
        time.sleep(0.01)  # Small sleep to prevent busy waiting
    
    thread.join()
    print("Done")

if __name__ == "__main__":
    main()