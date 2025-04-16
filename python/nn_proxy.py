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
    
    # Prefetch some tensors to avoid memory allocation during inference
    input_tensors = {}
    
    # Precompute batch sizes
    optimal_batch_sizes = [1, 8, 16, 32, 64, 128, 256]
    
    debug_print("Model worker ready to process requests")
    
    # Main worker loop
    while is_running:
        try:
            # Get batch of requests (non-blocking with timeout)
            requests = []
            try:
                # Short timeout to remain responsive
                requests.append(request_queue.get(timeout=0.005))
                request_queue.task_done()  # Mark as done immediately to avoid deadlocks
                
                # Get current queue size for stats
                current_size = request_queue.qsize()
                if current_size > 0:
                    debug_print(f"Current queue size: {current_size}")
                
                # Quickly collect more requests in current batch if available
                start_collection = time.time()
                collection_timeout = 0.002  # 2ms max collection time
                
                # Determine target batch size based on queue size
                target_size = 1
                for size in optimal_batch_sizes:
                    if current_size >= size:
                        target_size = size
                
                # Cap at MAX_BATCH_SIZE
                target_size = min(target_size, MAX_BATCH_SIZE)
                
                # Collect up to target_size or until collection timeout
                while len(requests) < target_size:
                    try:
                        # Use non-blocking get with very short timeout
                        req = request_queue.get(block=False)
                        requests.append(req)
                        request_queue.task_done()  # Mark as done immediately
                        
                        # Check if we've spent too long collecting
                        if time.time() - start_collection > collection_timeout:
                            break
                    except queue.Empty:
                        break
                
                actual_batch_size = len(requests)
                if actual_batch_size > 1:
                    debug_print(f"Collected batch of {actual_batch_size} requests in {(time.time() - start_collection)*1000:.1f}ms")
                
            except queue.Empty:
                # No requests available, just continue loop
                time.sleep(0.001)  # Short sleep to prevent CPU spinning
                continue
            
            # Start timing the batch processing
            batch_start = time.time()
            
            # Process batch with model
            try:
                results = process_batch_with_model(requests, scaler, device)
                
                # Calculate elapsed time
                batch_time = time.time() - batch_start
                
                # Update statistics
                total_batches += 1
                total_samples += len(requests)
                total_time += batch_time
                
                # Log performance for medium to large batches
                if len(requests) > 8:
                    debug_print(f"Batch of {len(requests)} processed in {batch_time:.3f}s ({batch_time/len(requests)*1000:.1f}ms per sample)")
                
                # Periodically log overall statistics
                if total_batches % 20 == 0:
                    avg_time = total_time/total_batches if total_batches > 0 else 0
                    avg_per_sample = total_time/total_samples if total_samples > 0 else 0
                    debug_print(f"Stats: {total_batches} batches, {total_samples} samples, "
                              f"avg {avg_time:.3f}s per batch, {avg_per_sample*1000:.1f}ms per sample")
                
                # Return results
                for i, result in enumerate(results):
                    if i < len(requests):
                        request_id = requests[i][0]
                        response_queue.put((request_id, result))
            
            except Exception as e:
                debug_print(f"Error processing batch: {str(e)}")
                debug_print(traceback.format_exc())
                
                # Create default responses for all requests in batch
                for req in requests:
                    try:
                        request_id = req[0]
                        # Default policy (uniform) and value (0)
                        default_result = (np.ones(225)/225, 0.0)
                        response_queue.put((request_id, default_result))
                        debug_print(f"Sent default response for request {request_id} after error")
                    except Exception as err:
                        debug_print(f"Error creating default response: {str(err)}")
        
        except Exception as e:
            debug_print(f"Error in model worker main loop: {str(e)}")
            debug_print(traceback.format_exc())
            time.sleep(0.01)  # Brief sleep after error

    debug_print("Model worker thread exiting")

def process_batch_with_model(requests, scaler=None, device="cuda"):
    """
    Process a batch of requests with the neural network model.
    Uses mixed precision for faster inference when available.
    
    Args:
        requests: List of (request_id, state_str, chosen_move, attack, defense) tuples
        scaler: GradScaler for mixed precision if enabled
        device: Device to run inference on
    
    Returns:
        List of (policy, value) tuples
    """
    global model_instance, USE_MIXED_PRECISION
    
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
        
        # Create input tensor with appropriate precision
        dtype = torch.float16 if USE_MIXED_PRECISION and device == "cuda" else torch.float32
        x_input = np.zeros((batch_size, input_dim), dtype=np.float32)
        
        # Process each input
        parse_start = time.time()
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
                    if prev_move >= 0 and j < num_history_moves:
                        x_input[i, offset + j] = float(prev_move) / (bs*bs)
                
                # Add previous moves for opponent
                offset = bs*bs + 1 + num_history_moves
                for j, prev_move in enumerate(opponent_moves_list[:num_history_moves]):
                    if prev_move >= 0 and j < num_history_moves:
                        x_input[i, offset + j] = float(prev_move) / (bs*bs)
                
                # Add attack and defense scores (normalized)
                x_input[i, -2] = min(max(attack, -1.0), 1.0)  # Clamp to [-1, 1]
                x_input[i, -1] = min(max(defense, -1.0), 1.0)  # Clamp to [-1, 1]
                
            except Exception as e:
                debug_print(f"Error parsing input {i}: {str(e)}")
                # Continue with zeros for this input
        
        parse_time = time.time() - parse_start
        if batch_size > 1:
            debug_print(f"Input parsing completed in {parse_time:.3f}s")
        
        # Convert to PyTorch tensor
        if batch_size > 16:
            debug_print(f"Created input tensor with shape {x_input.shape}")
        
        t_input = torch.tensor(x_input, dtype=dtype, device=torch.device(device))
        
        # Run forward pass with mixed precision if enabled
        with torch.no_grad():
            start_forward = time.time()
            
            try:
                if USE_MIXED_PRECISION and device == "cuda":
                    # Use autocast for mixed precision inference
                    try:
                        # New PyTorch 2.0+ style
                        with torch.amp.autocast('cuda'):
                            policy_logits, value_out = model_instance(t_input)
                    except TypeError:
                        # Fallback for older PyTorch
                        with torch.cuda.amp.autocast():
                            policy_logits, value_out = model_instance(t_input)
                else:
                    # Regular inference
                    policy_logits, value_out = model_instance(t_input)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    # Handle CUDA OOM by clearing cache and retrying with reduced precision
                    debug_print("CUDA OOM detected, clearing cache and retrying with reduced precision")
                    torch.cuda.empty_cache()
                    
                    # Force float16 for OOM recovery
                    t_input = t_input.half()
                    
                    # Retry with reduced precision
                    with torch.cuda.amp.autocast():
                        policy_logits, value_out = model_instance(t_input)
                else:
                    # Re-raise other runtime errors
                    raise
            
            forward_time = time.time() - start_forward
            if batch_size > 1:
                debug_print(f"Forward pass completed in {forward_time:.3f}s")
            
            # Move results to CPU and convert to appropriate precision
            policy_logits = policy_logits.cpu().float()
            value_out = value_out.cpu().float()
        
        # Convert to probabilities using softmax
        policy_probs = F.softmax(policy_logits, dim=1).numpy()
        values = value_out.squeeze(-1).numpy()
        
        # Build output
        results = []
        for i in range(batch_size):
            policy = policy_probs[i].tolist()
            value = float(values[i])
            results.append((policy, value))
        
        if batch_size > 16:
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