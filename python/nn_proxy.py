# python/nn_proxy.py - Fixed robust version to prevent segmentation faults

import queue
import threading
import time
import sys
import torch
import torch.nn.functional as F
import numpy as np
import re
import traceback
import weakref
import gc

# Global variables to hold the thread and queues
model_thread = None
request_queue = None
response_queue = None
model_instance = None
is_initialized = False
is_running = False
debug_mode = True  # Set to False in production

# Keep a weak reference to the module to detect when it's being garbage collected
nn_proxy_self = None

# Constants
MAX_BATCH_SIZE = 256  # Maximum batch size for better GPU utilization
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
    global model_thread, request_queue, response_queue, model_instance, is_initialized, is_running, MAX_BATCH_SIZE, USE_MIXED_PRECISION, nn_proxy_self
    
    # Force garbage collection first to clean up any previous resources
    gc.collect()
    
    # Store a weak reference to this module
    nn_proxy_self = weakref.ref(sys.modules[__name__])
    
    # Handle case where initialize is called again
    if is_initialized or is_running:
        debug_print("Neural network proxy already initialized, shutting down first")
        try:
            shutdown()
            time.sleep(0.1)  # Brief pause to ensure cleanup
        except Exception as e:
            debug_print(f"Error during reinitialization shutdown: {e}")
    
    debug_print(f"Initializing neural network proxy with batch_size={batch_size}, device={device}")
    
    # Clear thread and queue references
    model_thread = None
    
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
    request_queue = queue.Queue(maxsize=10000)  # Increased from default
    response_queue = queue.Queue(maxsize=10000)
    
    # Mark as initialized and running
    is_initialized = True
    is_running = True
    
    # Force a garbage collection cycle to clean up any previous resources
    gc.collect()
    
    # Start dedicated model thread
    model_thread = threading.Thread(target=model_worker, args=(device,), daemon=True)
    model_thread.start()
    
    debug_print(f"Neural network proxy initialized successfully with batch size {MAX_BATCH_SIZE}")
    debug_print(f"Inference configuration: device={device}, mixed_precision={USE_MIXED_PRECISION}")
    
    # Wait a brief moment to ensure thread has started
    time.sleep(0.1)
    
    return True

def shutdown():
    """Shutdown the neural network proxy system"""
    global is_running, is_initialized, request_queue, response_queue, model_instance, model_thread
    
    debug_print("Shutting down neural network proxy")
    
    # First mark as not initialized to prevent new requests
    is_initialized = False
    is_running = False
    
    # Clear any circular references
    model_copy = model_instance
    model_instance = None
    
    # Empty the queues to avoid deadlocks
    try:
        if request_queue:
            try:
                while not request_queue.empty():
                    try:
                        request_queue.get_nowait()
                        request_queue.task_done()
                    except:
                        pass
            except:
                pass
    except:
        pass
    
    try:
        if response_queue:
            try:
                while not response_queue.empty():
                    try:
                        response_queue.get_nowait()
                        response_queue.task_done()
                    except:
                        pass
            except:
                pass
    except:
        pass
    
    # Don't wait for thread to exit - let Python clean it up
    model_thread = None
    
    # Wait briefly to allow resources to be released
    time.sleep(0.05)
    
    # Clear the queues after operations
    request_queue = None
    response_queue = None
    
    # Force release CUDA memory if using it
    if model_copy is not None and hasattr(model_copy, 'to'):
        try:
            # Force model to CPU to release GPU memory
            model_copy.to('cpu')
            model_copy = None
            
            # Clear any CUDA caches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Force garbage collection
            gc.collect()
        except:
            debug_print("Error releasing model memory")
    
    debug_print("Neural network proxy shutdown complete")

def model_worker(device="cuda"):
    """Worker function that runs in a dedicated thread and owns the neural network model"""
    global model_instance, is_running, USE_MIXED_PRECISION, is_initialized
    
    debug_print(f"Model worker thread starting on device: {device}")
    
    # Move model to the appropriate device
    if model_instance is not None:
        try:
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
        except Exception as e:
            debug_print(f"Error setting up model: {str(e)}")
            is_initialized = False
            is_running = False
            return
    else:
        debug_print("WARNING: No model instance available")
        is_initialized = False
        is_running = False
        return
    
    # Initialize mixed precision scaler if using mixed precision
    scaler = None
    if USE_MIXED_PRECISION and device == "cuda":
        try:
            # Use the new style for PyTorch 2.0+
            scaler = torch.amp.GradScaler('cuda')
        except TypeError:
            # Fallback for older PyTorch versions
            try:
                scaler = torch.cuda.amp.GradScaler()
            except:
                debug_print("Error creating GradScaler, disabling mixed precision")
                USE_MIXED_PRECISION = False
        
        if scaler is not None:
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
    
    # Main worker loop with health monitoring
    last_health_check = time.time()
    last_activity = time.time()
    
    while is_running:
        try:
            # Periodically check if the module is being garbage collected
            current_time = time.time()
            if current_time - last_health_check > 5.0:  # Check every 5 seconds
                last_health_check = current_time
                
                # Check if our module is still valid
                if nn_proxy_self is None or nn_proxy_self() is None:
                    debug_print("Module being garbage collected, exiting worker")
                    break
                
                # Check if model instance is still valid
                if model_instance is None:
                    debug_print("Model instance is None, exiting worker")
                    break
                
                # Check for inactivity timeout (30 seconds - reduced from 2 minutes)
                if current_time - last_activity > 30.0:
                    debug_print("Worker inactive for 30 seconds, exiting")
                    break
                
                # Check if queues are still valid
                if request_queue is None or response_queue is None:
                    debug_print("Queues are None, exiting worker")
                    break
            
            # Process any available requests with timeout handling
            try:
                # Check if there's anything to process without blocking
                if request_queue.empty():
                    # Short sleep to avoid tight loop
                    time.sleep(0.01)
                    continue
                
                # Get first request to start a batch
                request = request_queue.get_nowait()
                last_activity = time.time()  # Update activity timestamp
                
                # Process this request
                request_queue.task_done()
                requests = [request]
                
                # Try to get more requests up to batch size with minimal looping
                try:
                    # Get current queue size
                    try:
                        current_size = request_queue.qsize()
                    except:
                        current_size = 0
                    
                    # Use largest optimal batch size that's less than queue size
                    target_size = 1
                    for size in optimal_batch_sizes:
                        if current_size >= size and size <= MAX_BATCH_SIZE:
                            target_size = size
                    
                    # Get up to target_size total requests
                    remaining = max(0, target_size - len(requests))
                    
                    # Only try to get more if there are actually more available
                    if remaining > 0 and current_size > 0:
                        # Collect more items without risking deadlock
                        collection_start = time.time()
                        collection_timeout = 0.02  # 20ms max collection time
                        
                        while len(requests) < target_size:
                            # Check for timeout or module shutdown
                            if time.time() - collection_start > collection_timeout or not is_running:
                                break
                                
                            try:
                                req = request_queue.get_nowait()
                                requests.append(req)
                                request_queue.task_done()
                            except queue.Empty:
                                break
                except Exception as e:
                    debug_print(f"Error collecting batch: {str(e)}")
                
                # Process the batch if we have any requests
                if requests and is_running:
                    batch_start = time.time()
                    
                    try:
                        results = process_batch_with_model(requests, scaler, device, input_cache)
                        
                        # Return results to appropriate callers
                        for i, result in enumerate(results):
                            if i < len(requests):
                                request_id = requests[i][0]
                                try:
                                    # Use put_nowait to avoid blocking
                                    response_queue.put_nowait((request_id, result))
                                except queue.Full:
                                    debug_print(f"Response queue full, dropping result for request {request_id}")
                        
                        # Update statistics
                        batch_time = time.time() - batch_start
                        total_batches += 1
                        total_samples += len(requests)
                        total_time += batch_time
                        
                        # Log performance for medium to large batches
                        if len(requests) > 8:
                            debug_print(f"Batch of {len(requests)} processed in {batch_time:.3f}s ({batch_time/len(requests)*1000:.1f}ms per sample)")
                    except Exception as e:
                        debug_print(f"Error processing batch: {str(e)}")
                        traceback.print_exc()
                        
                        # Return default values on error
                        for req in requests:
                            try:
                                request_id = req[0]
                                # Create simple uniform policy for default
                                default_policy = np.ones(225)/225
                                default_value = 0.0
                                response_queue.put_nowait((request_id, (default_policy, default_value)))
                            except:
                                pass
            except queue.Empty:
                # No requests available, just continue loop
                pass
            except Exception as e:
                debug_print(f"Error processing request: {str(e)}")
                # Short sleep to avoid tight loop on repeated errors
                time.sleep(0.01)
        
        except Exception as e:
            debug_print(f"Error in model worker main loop: {str(e)}")
            # Short sleep to avoid tight loop on repeated errors
            time.sleep(0.01)
    
    debug_print("Model worker thread exiting")
    
    # Final cleanup
    try:
        # Release model memory
        if model_instance is not None:
            # Clear reference first
            temp_model = model_instance
            model_instance = None
            
            # Move to CPU to free GPU memory
            if device == "cuda":
                try:
                    temp_model.to("cpu")
                except:
                    pass
                
                # Force CUDA cache clear if available
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except:
                        pass
            
            # Release the reference
            del temp_model
        
        # Mark as not initialized
        is_initialized = False
        
        # Force garbage collection
        gc.collect()
        
    except Exception as e:
        debug_print(f"Error during worker cleanup: {str(e)}")

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
        
        # Convert to PyTorch tensor safely
        try:
            # Check if module and model are still valid
            if model_instance is None or not is_running:
                raise RuntimeError("Model no longer available")
            
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
            debug_print(f"Error during tensor processing: {e}")
            raise # Re-raise to be caught by outer handler
        
    except Exception as e:
        debug_print(f"Error in process_batch_with_model: {e}")
        debug_print(traceback.format_exc())
        # Return default values
        return [(np.ones(board_size*board_size)/float(board_size*board_size), 0.0) for _ in requests]

def get_request_info():
    """Get information about the request queue (for debugging)"""
    if request_queue is None:
        return "Request queue not initialized"
    
    try:
        return f"Request queue size: {request_queue.qsize()}"
    except:
        return "Error getting request queue size"

def get_response_info():
    """Get information about the response queue (for debugging)"""
    if response_queue is None:
        return "Response queue not initialized"
    
    try:
        return f"Response queue size: {response_queue.qsize()}"
    except:
        return "Error getting response queue size"

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
                response_queue.task_done()  # Mark as done
                
                if resp_id == request_id:
                    return result
                else:
                    # Put it back if it's not ours
                    response_queue.put((resp_id, result))
            
            # Brief sleep
            time.sleep(0.01)
        except queue.Empty:
            pass
    
    # Timeout - return default instead of raising exception
    default_policy = np.ones(225)/225
    default_value = 0.0
    debug_print(f"Inference request timed out after {DEFAULT_TIMEOUT} seconds, returning default values")
    return (default_policy.tolist(), default_value)
