# **Gomoku MCTS AI with Attack/Defense – Product Requirements Document (PRD)**

## **1. Introduction**

This document outlines the requirements for a **Gomoku MCTS AI** project that:
- Implements **variable-size Gomoku** logic (e.g. 9×9, 15×15, 19×19).
- Incorporates **Attack/Defense scoring** for enhanced decision-making.
- Uses a **multi-threaded MCTS** algorithm in C++.
- Bridges to **Python** (via **pybind11**) for self-play, training data collection, and neural network inference.
- Demonstrates an **example PyTorch** model that accepts `(board state + attack/defense + chosen move)` as input.

The goal is a maintainable, high-performance system for Gomoku AI research and training.

---

## **2. Project Overview and Objectives**

1. **Objective**: Provide a robust MCTS-based Gomoku AI that leverages domain-specific Attack/Defense bonuses, with flexible board sizes and Python integration for deep learning training.
2. **Key Features**:
   - **Multi-threaded MCTS** with concurrency safety.
   - **Attack/Defense** logic from existing `attack_defense.cpp/h`.
   - **Gomoku** logic from existing `gomoku.cpp/h`.
   - **Python** scripts for self-play data generation (training) and arena (AI vs. AI matches).
   - A basic **PyTorch** model to show integration with GPU-based inference.

---

## **3. Folder / File Structure**

Below is the recommended layout to keep the project organized:

```
mcts_project/
├── CMakeLists.txt                  # (Optional if using CMake)
├── setup.py                       # For pip-based installation of the C++ extension
├── include/
│   ├── mcts_config.h              # MCTS parameter struct
│   ├── node.h                     # MCTS Node class (thread-safe stats)
│   ├── mcts.h                     # Main MCTS engine (multi-threaded)
│   ├── nn_interface.h             # BatchingNNInterface for batched GPU inference
│   ├── gomoku.h                   # Original Gomoku logic (unchanged)
│   ├── attack_defense.h           # Original Attack/Defense logic (unchanged)
├── src/
│   ├── mcts_config.cpp
│   ├── node.cpp
│   ├── mcts.cpp
│   ├── nn_interface.cpp
│   ├── gomoku.cpp                 # Original Gomoku logic (unchanged)
│   ├── attack_defense.cpp         # Original Attack/Defense logic (unchanged)
│   └── python_wrapper.cpp         # PyBind11-based module (MCTSWrapper etc.)
└── python/
    ├── model.py                   # Example PyTorch model
    ├── self_play.py               # Self-play data generation
    ├── arena.py                   # Arena matches between two agents
    └── example_train_script.py    # Illustrative training approach
```

**Notes**:  
- The **`gomoku.cpp/h`** and **`attack_defense.cpp/h`** files remain **“as is”** per the project’s requirement.  
- The **C++** MCTS code references these for board rules (Gomoku) and domain scoring (AttackDefense).  
- **Python** scripts rely on the compiled extension (`mcts_py`) installed via `setup.py`.

---

## **4. Functional Requirements**

### **4.1. Gomoku Logic**
1. **Variable Board Size**: Must support NxN board (e.g., 9, 15, 19, etc.) configured at runtime.  
2. **Forbidden Moves** (Renju) / **Overlines** (Omok) if flags are set.  
3. **Key Methods** (in `Gamestate` or similarly named class):
   - `make_move(action, player)`
   - `is_terminal()`
   - `get_winner()`
   - `get_valid_moves()`
   - `copy()`
   - Possibly `get_board()` for AttackDefense integration.

### **4.2. AttackDefense Logic**
1. **Use**: `attack_defense.cpp/h` as originally provided.  
2. **`compute_bonuses(...)`**: Takes a **batch** of boards, moves, players -> outputs `(attackVec, defenseVec)`.  
3. For single-state usage, MCTS can form a batch of size 1.

### **4.3. MCTS Engine**
1. **Multi-threading**:
   - Launch `config.num_threads` worker threads.  
   - Each performs selection -> expansion -> evaluation -> backup until `num_simulations` is reached.  
2. **Concurrency-Safe Node**:
   - Atomic `visit_count_` and `total_value_`.  
   - Per-node expand mutex to prevent double expansion.  
3. **Integration**:
   - AttackDefense: For each leaf’s `(Gamestate, chosenMove)`, we compute `(attack, defense)`.  
   - BatchingNNInterface: We pass `(Gamestate, chosenMove, attack, defense)` to the NN in one batch call.

### **4.4. BatchingNNInterface**
1. **Queue** of requests `(Gamestate, chosenMove, attack, defense)`.  
2. A **background worker**:
   - Waits on condition variable, collects all pending requests in one go, calls Python once to do the GPU inference.  
   - Returns `(policy, value)` results to each request.  

### **4.5. Python Binding and Scripts**
1. **`python_wrapper.cpp`**:
   - A `MCTSWrapper` class with constructor `(MCTSConfig, boardSize, useRenju, useOmok, seed, useProLongOpening, ...)`.  
   - Methods: `set_infer_function(...)`, `run_search()`, `best_move()`, `apply_best_move()`, `is_terminal()`, `get_winner()`.  
2. **Self-play** (`self_play.py`):
   - Repeatedly call `run_search()` -> `best_move()` -> `apply_best_move()` until terminal, capturing moves for training data.  
3. **Arena** (`arena.py`):
   - Two MCTSWrappers, each with possibly different nets, alternate moves.  
4. **Model** (`model.py`):
   - Minimal PyTorch net that processes board data and `(attack, defense, chosenMove)`.  
5. **Training** (`example_train_script.py`):
   - Illustrates how to load self-play data, train the net, and store weights.

---

## **5. Non-Functional Requirements**

1. **Concurrency Safety**:
   - No data races in node expansions or stats updates.  
   - Single background inference thread in `BatchingNNInterface`.  
2. **Performance**:
   - MCTS should scale with CPU cores.  
   - The GPU-based NN calls happen in large batches for efficiency.  
3. **Maintainability**:
   - `gomoku.cpp/h` and `attack_defense.cpp/h` remain unmodified.  
   - MCTS references them via well-defined methods.

---

## **6. Success Criteria**

1. **Compilation & Installation**:  
   - `pip install .` successfully builds `mcts_py` extension.  
2. **Variable Board Size**:  
   - The user can pass different board sizes to `MCTSWrapper` without code changes.  
3. **Attack/Defense**:  
   - `(attack, defense)` is computed for each chosen move in MCTS expansions.  
   - The neural net callback receives these bonuses and can integrate them in predictions.  
4. **Python Self-play**:
   - `self_play.py` runs to completion, generating valid moves and storing data.  
5. **Arena**:
   - `arena.py` can run multiple matches between two MCTS-based players with different nets.  

---

## **7. Technical Implementation Summary**

1. **Core C++**:
   - **`gomoku.cpp/h`**: Original Gomoku logic (with or without Renju/Omok rules).  
   - **`attack_defense.cpp/h`**: Domain-specific scoring.  
   - **MCTS**:
     - A `Node` class with atomic counters.  
     - `MCTS` class that spawns threads and does parallel rollouts.  
     - For each leaf, calls AttackDefense (batch=1) + BatchingNNInterface.  
   - **`BatchingNNInterface`** to unify GPU calls in Python.  
2. **Python**:
   - `MCTSWrapper` bridging to the C++ MCTS.  
   - Scripts to handle self-play, data saving, and arena matches.  
   - A minimal PyTorch model that demonstrates how to parse the `(attack, defense)` input.  

---

## **8. Implementation Plan**

1. **Phase 1**: Wrap `gomoku.cpp/h` and `attack_defense.cpp/h` in C++ MCTS logic (with concurrency).  
2. **Phase 2**: Provide a pybind11 interface (`python_wrapper.cpp`) and minimal test script.  
3. **Phase 3**: Add the `BatchingNNInterface` background worker for GPU inference.  
4. **Phase 4**: Write `self_play.py`, `arena.py`, `model.py` for demonstration.  
5. **Phase 5** (Optional): Integrate a real training pipeline using `example_train_script.py`.

---

## **9. Risks / Mitigations**

1. **Concurrency**: 
   - Potential race conditions. Mitigate with atomic counters and minimal locks.  
2. **Large boards**: 
   - High compute cost. Possibly reduce `num_simulations` or optimize.  
3. **Complex Attack/Defense** logic**: 
   - Must ensure it is called consistently (batch size=1 or more) to avoid overhead.  

---

## **10. Conclusion**

This PRD specifies the final structure, features, and flow for a **Gomoku MCTS** system that integrates **Attack/Defense** scoring and Python-based neural inference. Following the architecture laid out above ensures:

- **Clean separation** of domain logic (Gomoku, AttackDefense) from MCTS concurrency.  
- **Scalability** across different board sizes and CPU/GPU resources.  
- **Easy extensibility** for advanced training or multi-agent competitions.