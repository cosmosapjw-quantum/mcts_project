# python/nn_proxy.py - Optimized version for GPU utilization

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

# Constants - OPTIMIZED FOR 3060 Ti 8GB
MAX_BATCH_SIZE = 256  # Increased from 16 to 256 for better GPU utilization
DEFAULT_TIMEOUT = 1.0  # seconds
USE_MIXED_PRECISION = True  # Use FP16 for faster inference

def debug_print(message):
    """Print a debug message with timestamp"""
    if debug_mode:
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        print(f"[PYMODEL {timestamp}] {message}", flush=True)

def initialize_neural_network(model, batch_size=256, device="cuda"):
    """
    Initialize the neural network proxy system with optimizations for GPU.
    
    Args:
        model: The neural network model instance
        batch_size: Maximum batch size for evaluation (default: 256 for better GPU utilization)
        device: Device to run the model on ('cuda' or 'cpu')
    """
    global model_thread, request_queue, response_queue, model_instance, is_initialized, is_running, MAX_BATCH_SIZE, USE_MIXED_PRECISION
    
    # Only initialize once
    if is_initialized:
        debug_print("Neural network proxy already initialized")
        return
    
    debug_print(f"Initializing neural network proxy with batch_size={batch_size}, device={device}")
    
    # Configure batch size based on device and available memory
    if device == "cuda" and torch.cuda.is_available():
        # Get GPU properties
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        
        debug_print(f"Detected GPU: {gpu_name} with {gpu_memory:.1f}GB VRAM")
        
        # Optimize batch size based on GPU model for RTX 3060 Ti
        if "3060 Ti" in gpu_name and batch_size < 256:
            batch_size = 256
            debug_print(f"Automatically increased batch size to {batch_size} for {gpu_name}")
        
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
        debug_print("Enabled cuDNN benchmark for performance optimization")
        
        # Enable mixed precision (FP16) for faster inference
        USE_MIXED_PRECISION = True
        debug_print("Mixed precision (FP16) enabled for faster inference")
            
    else:
        # If using CPU, use smaller batches
        batch_size = min(64, batch_size)
        USE_MIXED_PRECISION = False
        debug_print(f"Using CPU mode with batch size {batch_size}")
    
    # Store the model
    model_instance = model
    MAX_BATCH_SIZE = batch_size
    
    # Create thread-safe queues with increased size limits
    request_queue = queue.Queue(maxsize=10000)  # Increased from default to handle more requests
    response_queue = queue.Queue(maxsize=10000)
    
    # Mark as initialized and running
    is_initialized = True
    is_running = True
    
    # Start dedicated model thread
    model_thread = threading.Thread(target=model_worker, args=(device,), daemon=True)
    model_thread.start()
    
    debug_print(f"Neural network proxy initialized successfully with batch size {MAX_BATCH_SIZE}")
    debug_print(f"Inference configuration: device={device}, mixed_precision={USE_MIXED_PRECISION}")
    
    # Wait a brief moment to ensure thread has started
    time.sleep(0.1)

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
    global model_instance, is_running, USE_MIXED_PRECISION
    
    debug_print(f"Model worker thread starting on device: {device}")
    
    # Move model to the appropriate device
    if model_instance is not None:
        model_instance.to(torch.device(device))
        model_instance.eval()  # Set to evaluation mode
        
        # Enable CUDA optimizations for better performance
        if device == "cuda" and torch.cuda.is_available():
            # Use cudnn benchmarking for optimized convolution algorithms
            torch.backends.cudnn.benchmark = True
            
            # Print model summary
            param_count = sum(p.numel() for p in model_instance.parameters())
            debug_print(f"Model has {param_count:,} parameters")
            debug_print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB total")
            debug_print(f"Mixed precision: {USE_MIXED_PRECISION}")
    else:
        debug_print("WARNING: No model instance available")
        return
    
    # Initialize mixed precision scaler if using mixed precision
    scaler = None
    if USE_MIXED_PRECISION and device == "cuda":
        try:
            # Use the new style for PyTorch 2.0+
            scaler = torch.amp.GradScaler('cuda')
        except TypeError:
            # Fallback for older PyTorch versions
            scaler = torch.cuda.amp.GradScaler()
        debug_print("Created GradScaler for mixed precision")
    
    # Statistics
    total_batches = 0
    total_samples = 0
    total_time = 0
    
    # Precompute optimal batch sizes
    optimal_batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256]
    
    # Cache common tensor shapes for reuse
    input_cache = {}
    
    debug_print("Model worker ready to process requests")
    
    # Main worker loop - use adaptive polling based on queue size
    last_queue_check = time.time()
    empty_queue_sleep = 0.001  # 1ms when queue is empty
    
    while is_running:
        try:
            # Check if there are any requests in the queue without blocking
            queue_empty = True
            try:
                queue_empty = request_queue.empty()
            except:
                pass
            
            # If queue is empty, sleep briefly and check again
            if queue_empty:
                # Progressive backoff for empty queue
                current_time = time.time()
                if current_time - last_queue_check > 1.0:  # Check queue size once per second max
                    try:
                        current_size = request_queue.qsize()
                        if current_size > 0:
                            debug_print(f"Queue has {current_size} requests after empty check")
                            queue_empty = False
                        last_queue_check = current_time
                    except:
                        pass
                
                if queue_empty:
                    time.sleep(empty_queue_sleep)
                    # Gradually increase sleep time if queue remains empty
                    empty_queue_sleep = min(0.01, empty_queue_sleep * 1.2)  # Cap at 10ms
                    continue
            else:
                # Reset sleep time when queue has items
                empty_queue_sleep = 0.001
                last_queue_check = time.time()
            
            # Get batch of requests (non-blocking with timeout)
            requests = []
            try:
                # Short timeout to remain responsive
                requests.append(request_queue.get(timeout=0.005))
                request_queue.task_done()  # Mark as done immediately to avoid deadlocks
                
                # Determine target batch size based on queue size
                try:
                    current_size = request_queue.qsize()
                    target_size = 1
                    
                    # Use largest optimal batch size that's less than queue size
                    for size in optimal_batch_sizes:
                        if current_size >= size:
                            target_size = size
                    
                    # Cap at MAX_BATCH_SIZE
                    target_size = min(target_size, MAX_BATCH_SIZE)
                    
                    if current_size > 32:
                        debug_print(f"Queue has {current_size} requests, target batch size: {target_size}")
                except:
                    target_size = min(16, MAX_BATCH_SIZE)  # Conservative default
                
                # Collect batch with adaptive timeout
                batch_collect_start = time.time()
                batch_timeout = 0.001 * min(20, max(1, target_size // 8))  # Longer timeout for larger batches
                
                # Collect up to target_size or until timeout
                while len(requests) < target_size:
                    try:
                        req = request_queue.get(block=False)
                        requests.append(req)
                        request_queue.task_done()
                        
                        # Check timeout - shorter timeout for smaller batches
                        if time.time() - batch_collect_start > batch_timeout:
                            break
                    except queue.Empty:
                        break
                
                batch_size = len(requests)
                if batch_size > 16:
                    collect_time = (time.time() - batch_collect_start) * 1000
                    debug_print(f"Collected batch of {batch_size}/{target_size} in {collect_time:.1f}ms")
                
            except queue.Empty:
                # No requests available, just continue loop
                continue
            
            # Process batch with model
            try:
                batch_start = time.time()
                results = process_batch_with_model(requests, scaler, device, input_cache)
                batch_time = time.time() - batch_start
                
                # Update statistics
                total_batches += 1
                total_samples += len(requests)
                total_time += batch_time
                
                # Log performance for medium to large batches
                if len(requests) > 8:
                    debug_print(f"Batch of {len(requests)} processed in {batch_time:.3f}s ({batch_time/len(requests)*1000:.1f}ms per sample)")
                
                # Periodically log overall statistics
                if total_batches % 50 == 0:
                    avg_time = total_time/total_batches if total_batches > 0 else 0
                    avg_per_sample = total_time/total_samples if total_samples > 0 else 0
                    debug_print(f"Stats: {total_batches} batches, {total_samples} samples, "
                              f"avg {avg_time:.3f}s per batch, {avg_per_sample*1000:.1f}ms per sample")
                
                # Return results without blocking
                for i, result in enumerate(results):
                    if i < len(requests):
                        request_id = requests[i][0]
                        try:
                            # Non-blocking put
                            response_queue.put((request_id, result), block=False)
                        except queue.Full:
                            debug_print(f"Response queue full, dropping result for request {request_id}")
            
            except Exception as e:
                debug_print(f"Error processing batch: {str(e)}")
                debug_print(traceback.format_exc())
                
                # Create default responses
                for req in requests:
                    try:
                        request_id = req[0]
                        default_result = (np.ones(225)/225, 0.0)
                        response_queue.put((request_id, default_result), block=False)
                    except:
                        pass
        
        except Exception as e:
            debug_print(f"Error in model worker main loop: {str(e)}")
            time.sleep(0.01)

    debug_print("Model worker thread exiting")

def process_batch_with_model(requests, scaler=None, device="cuda", input_cache=None):
    """
    Process a batch of requests with optimized tensor handling and caching
    """
    global model_instance, USE_MIXED_PRECISION
    
    if model_instance is None:
        return [(np.ones(225)/225, 0.0) for _ in requests]
    
    # Extract state dimensions from first request
    batch_size = len(requests)
    board_size = 15  # Default
    num_history_moves = getattr(model_instance, 'num_history_moves', 3)
    
    try:
        # Parse first state to get board size
        if batch_size > 0:
            first_state = requests[0][1]  # state_str
            board_size_match = re.search(r'Board:(\d+)', first_state)
            if board_size_match:
                board_size = int(board_size_match.group(1))
    except:
        pass
    
    # Calculate input dimension
    input_dim = board_size*board_size + 1 + 2*num_history_moves + 2
    
    # Use cached tensors when possible
    tensor_key = f"{batch_size}_{input_dim}"
    x_input = None
    
    if input_cache is not None and tensor_key in input_cache:
        # Reuse cached tensor (zeroing it first)
        x_input = input_cache[tensor_key]
        x_input.fill(0)
    else:
        # Create new tensor
        x_input = np.zeros((batch_size, input_dim), dtype=np.float32)
        # Cache for future use (up to some reasonable limit)
        if input_cache is not None and len(input_cache) < 20:
            input_cache[tensor_key] = x_input
    
    # Parse inputs in parallel with numpy operations when possible
    try:
        # Process each input - this is the performance-critical section
        for i, (_, state_str, chosen_move, attack, defense) in enumerate(requests):
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
                    elif key in ['Board', 'Player']:
                        board_info[key] = value
            
            # Get the board size and current player
            bs = int(board_info.get('Board', str(board_size)))
            current_player = int(board_info.get('Player', '1'))
            
            # Fill the board array from the state string
            if state_string and len(state_string) == bs*bs:
                for j, c in enumerate(state_string):
                    cell_value = int(c)
                    if cell_value == current_player:
                        x_input[i, j] = 1.0  # Current player's stone
                    elif cell_value != 0:
                        x_input[i, j] = -1.0  # Opponent's stone
            
            # Add player flag (1.0 for player 1, 0.0 for player 2)
            x_input[i, bs*bs] = 1.0 if current_player == 1 else 0.0
            
            # Add previous moves for current player (normalize positions)
            offset = bs*bs + 1
            for j, prev_move in enumerate(current_moves_list[:num_history_moves]):
                if prev_move >= 0 and j < num_history_moves:
                    x_input[i, offset + j] = float(prev_move) / (bs*bs)
            
            # Add previous moves for opponent
            offset = bs*bs + 1 + num_history_moves
            for j, prev_move in enumerate(opponent_moves_list[:num_history_moves]):
                if prev_move >= 0 and j < num_history_moves:
                    x_input[i, offset + j] = float(prev_move) / (bs*bs)
            
            # Add attack and defense scores
            x_input[i, -2] = min(max(attack, -1.0), 1.0)
            x_input[i, -1] = min(max(defense, -1.0), 1.0)
        
        # Convert to PyTorch tensor with appropriate precision
        dtype = torch.float16 if USE_MIXED_PRECISION and device == "cuda" else torch.float32
        t_input = torch.tensor(x_input, dtype=dtype, device=torch.device(device))
        
        # Run forward pass with mixed precision if enabled
        with torch.no_grad():
            if USE_MIXED_PRECISION and device == "cuda":
                with torch.amp.autocast('cuda'):
                    policy_logits, value_out = model_instance(t_input)
            else:
                policy_logits, value_out = model_instance(t_input)
            
            # Move results to CPU and convert to appropriate precision
            policy_probs = F.softmax(policy_logits, dim=1).cpu().float().numpy()
            values = value_out.cpu().float().squeeze(-1).numpy()
        
        # Build output
        results = []
        for i in range(batch_size):
            policy = policy_probs[i].tolist()
            value = float(values[i])
            results.append((policy, value))
        
        return results
        
    except Exception as e:
        debug_print(f"Error in process_batch_with_model: {e}")
        debug_print(traceback.format_exc())
        # Return default values
        return [(np.ones(board_size*board_size)/float(board_size*board_size), 0.0) for _ in requests]

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