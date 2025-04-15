# python/nn_proxy.py
import queue
import threading
import time
import sys
import torch
import torch.nn.functional as F
import numpy as np
import re
import traceback

# Global variables to hold the thread and queues
model_thread = None
request_queue = None
response_queue = None
model_instance = None
is_initialized = False
is_running = False
debug_mode = True  # Set to False in production

# Constants
MAX_BATCH_SIZE = 16
DEFAULT_TIMEOUT = 1.0  # seconds

def debug_print(message):
    """Print a debug message with timestamp"""
    if debug_mode:
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        print(f"[PYMODEL {timestamp}] {message}", flush=True)

def initialize_neural_network(model, batch_size=16, device="cuda"):
    """
    Initialize the neural network proxy system.
    
    Args:
        model: The neural network model instance
        batch_size: Maximum batch size for evaluation
        device: Device to run the model on ('cuda' or 'cpu')
    """
    global model_thread, request_queue, response_queue, model_instance, is_initialized, is_running, MAX_BATCH_SIZE
    
    # Only initialize once
    if is_initialized:
        debug_print("Neural network proxy already initialized")
        return
    
    debug_print(f"Initializing neural network proxy with batch_size={batch_size}, device={device}")
    
    # Store the model
    model_instance = model
    MAX_BATCH_SIZE = batch_size
    
    # Create thread-safe queues
    request_queue = queue.Queue()
    response_queue = queue.Queue()
    
    # Mark as initialized and running
    is_initialized = True
    is_running = True
    
    # Start dedicated model thread
    model_thread = threading.Thread(target=model_worker, args=(device,), daemon=True)
    model_thread.start()
    
    debug_print("Neural network proxy initialized successfully")

def shutdown():
    """Shutdown the neural network proxy system"""
    global is_running
    
    debug_print("Shutting down neural network proxy")
    is_running = False
    
    # Wait for thread to exit
    if model_thread is not None and model_thread.is_alive():
        try:
            model_thread.join(timeout=1.0)
            debug_print("Neural network thread joined successfully")
        except:
            debug_print("Timeout waiting for neural network thread to exit")
    
    debug_print("Neural network proxy shutdown complete")

def model_worker(device="cuda"):
    """Worker function that runs in a dedicated thread and owns the neural network model"""
    global model_instance, is_running
    
    debug_print(f"Model worker thread starting on device: {device}")
    
    # Move model to the appropriate device
    if model_instance is not None:
        model_instance.to(torch.device(device))
        model_instance.eval()  # Set to evaluation mode
    
    # Statistics
    total_batches = 0
    total_samples = 0
    total_time = 0
    
    # Main worker loop
    while is_running:
        try:
            # Get batch of requests (non-blocking with timeout)
            requests = []
            try:
                # Wait for at least one request
                requests.append(request_queue.get(timeout=0.01))  # Reduced from 0.1s to 0.01s
                
                # Collect any additional pending requests
                while len(requests) < MAX_BATCH_SIZE:
                    try:
                        requests.append(request_queue.get_nowait())
                    except queue.Empty:
                        break
                
                debug_print(f"Processing batch of {len(requests)} requests")
            except queue.Empty:
                # No requests available, just continue loop
                continue
            
            # Start timing the batch processing
            batch_start = time.time()
            
            # Process batch with model
            results = process_batch_with_model(requests)
            
            # Calculate elapsed time
            batch_time = time.time() - batch_start
            
            # Update statistics
            total_batches += 1
            total_samples += len(requests)
            total_time += batch_time
            
            # Log performance
            debug_print(f"Batch processed in {batch_time:.3f}s ({batch_time/len(requests):.3f}s per sample)")
            
            # Periodically log overall statistics
            if total_batches % 10 == 0:
                debug_print(f"Stats: {total_batches} batches, {total_samples} samples, "
                          f"avg {total_time/total_batches:.3f}s per batch")
            
            # Return results
            for i, result in enumerate(results):
                if i < len(requests):
                    request_id = requests[i][0]
                    response_queue.put((request_id, result))
                    request_queue.task_done()
        
        except Exception as e:
            debug_print(f"Error in model worker: {e}")
            debug_print(traceback.format_exc())
            time.sleep(0.01)  # Reduced from 0.1s to 0.01s

def process_batch_with_model(requests):
    """
    Process a batch of requests with the neural network model.
    
    Args:
        requests: List of (request_id, state_str, chosen_move, attack, defense) tuples
    
    Returns:
        List of (policy, value) tuples
    """
    global model_instance
    
    # Check if model is available
    if model_instance is None:
        debug_print("WARNING: Model is None, returning default values")
        return [(np.ones(225)/225, 0.0) for _ in requests]
    
    # Extract inputs from requests
    batch_size = len(requests)
    inputs = []
    
    for req_id, state_str, chosen_move, attack, defense in requests:
        inputs.append((state_str, chosen_move, attack, defense))
    
    # Apply a timeout to the entire processing
    try:
        # Parse inputs and create tensor
        start_time = time.time()
        
        # Get board size from first input
        board_size = 15  # default
        if inputs and len(inputs) > 0:
            first_state = inputs[0][0]  # Get first state string
            board_size_match = re.search(r'Board:(\d+)', first_state)
            if board_size_match:
                board_size = int(board_size_match.group(1))
        
        # Get history moves parameter from the model
        num_history_moves = getattr(model_instance, 'num_history_moves', 3)
        
        # Calculate input dimension
        input_dim = board_size*board_size + 1 + 2*num_history_moves + 2
        
        # Create input tensor
        x_input = np.zeros((batch_size, input_dim), dtype=np.float32)
        
        # Process each input
        for i, (state_str, chosen_move, attack, defense) in enumerate(inputs):
            try:
                # Parse the board state from state_str
                board_info = {}
                state_string = None
                current_moves_list = []
                opponent_moves_list = []
                
                # Split the string by semicolons
                parts = state_str.split(';')
                for part in parts:
                    if ':' in part:
                        key, value = part.split(':', 1)
                        if key == 'State':
                            state_string = value
                        elif key == 'CurrentMoves':
                            if value:
                                current_moves_list = [int(m) for m in value.split(',') if m]
                        elif key == 'OpponentMoves':
                            if value:
                                opponent_moves_list = [int(m) for m in value.split(',') if m]
                        elif key in ['Attack', 'Defense']:
                            # These are already provided as separate parameters
                            pass
                        else:
                            board_info[key] = value
                
                # Get the board size and current player
                bs = int(board_info.get('Board', str(board_size)))
                current_player = int(board_info.get('Player', '1'))
                
                # Create the board representation (flattened)
                board_array = np.zeros(bs*bs, dtype=np.float32)
                
                # Fill the board array from the state string
                if state_string and len(state_string) == bs*bs:
                    for j, c in enumerate(state_string):
                        cell_value = int(c)
                        if cell_value == current_player:
                            board_array[j] = 1.0  # Current player's stone
                        elif cell_value != 0:
                            board_array[j] = -1.0  # Opponent's stone (normalized to -1)
                
                # Fill the input tensor with board state
                x_input[i, :bs*bs] = board_array
                
                # Add player flag (1.0 for BLACK=1, 0.0 for WHITE=2)
                x_input[i, bs*bs] = 1.0 if current_player == 1 else 0.0
                
                # Add previous moves for current player (normalize positions)
                offset = bs*bs + 1
                for j, prev_move in enumerate(current_moves_list[:num_history_moves]):
                    if prev_move >= 0 and j < num_history_moves:  # Valid move and within history limit
                        # Normalize the move position
                        x_input[i, offset + j] = float(prev_move) / (bs*bs)
                
                # Add previous moves for opponent
                offset = bs*bs + 1 + num_history_moves
                for j, prev_move in enumerate(opponent_moves_list[:num_history_moves]):
                    if prev_move >= 0 and j < num_history_moves:  # Valid move and within history limit
                        # Normalize the move position
                        x_input[i, offset + j] = float(prev_move) / (bs*bs)
                
                # Add attack and defense scores (normalized)
                x_input[i, -2] = min(max(attack, -1.0), 1.0)  # Clamp to [-1, 1]
                x_input[i, -1] = min(max(defense, -1.0), 1.0)  # Clamp to [-1, 1]
                
            except Exception as e:
                debug_print(f"Error parsing input {i}: {str(e)}")
                # Continue with zeros for this input
        
        # Convert to PyTorch tensor
        debug_print(f"Input tensor created with shape {x_input.shape}")
        t_input = torch.from_numpy(x_input).to(next(model_instance.parameters()).device)
        
        # Run forward pass
        with torch.no_grad():
            debug_print("Running model forward pass")
            start_forward = time.time()
            policy_logits, value_out = model_instance(t_input)
            forward_time = time.time() - start_forward
            debug_print(f"Forward pass completed in {forward_time:.3f}s")
            
            # Move results to CPU
            policy_logits = policy_logits.cpu()
            value_out = value_out.cpu()
        
        # Convert to probabilities using softmax
        policy_probs = F.softmax(policy_logits, dim=1).numpy()
        values = value_out.squeeze(-1).numpy()
        
        # Build output
        results = []
        for i in range(batch_size):
            policy = policy_probs[i].tolist()
            value = float(values[i])
            results.append((policy, value))
        
        processing_time = time.time() - start_time
        debug_print(f"Total processing time: {processing_time:.3f}s")
        
        return results
        
    except Exception as e:
        debug_print(f"Error processing batch: {e}")
        debug_print(traceback.format_exc())
        # Return default values
        return [(np.ones(225)/225, 0.0) for _ in requests]

def get_request_info():
    """Get information about the request queue (for debugging)"""
    if request_queue is None:
        return "Request queue not initialized"
    
    return f"Request queue size: {request_queue.qsize()}"

def get_response_info():
    """Get information about the response queue (for debugging)"""
    if response_queue is None:
        return "Response queue not initialized"
    
    return f"Response queue size: {response_queue.qsize()}"

# Export a function for testing from Python
def test_inference(state_str, chosen_move, attack, defense):
    """
    Test the neural network inference from Python
    
    Args:
        state_str: Board state string
        chosen_move: Chosen move
        attack: Attack score
        defense: Defense score
    
    Returns:
        (policy, value) tuple
    """
    if not is_initialized:
        raise RuntimeError("Neural network proxy not initialized")
    
    # Generate a unique request ID
    request_id = int(time.time() * 1000) % 1000000
    
    # Add to request queue
    request_queue.put((request_id, state_str, chosen_move, attack, defense))
    
    # Wait for response with timeout
    start_time = time.time()
    while time.time() - start_time < DEFAULT_TIMEOUT:
        try:
            # Check for corresponding response
            if not response_queue.empty():
                resp_id, result = response_queue.get(block=False)
                if resp_id == request_id:
                    return result
                else:
                    # Put it back if it's not ours
                    response_queue.put((resp_id, result))
            
            # Brief sleep
            time.sleep(0.01)
        except queue.Empty:
            pass
    
    # Timeout
    raise TimeoutError(f"Inference request timed out after {DEFAULT_TIMEOUT} seconds")