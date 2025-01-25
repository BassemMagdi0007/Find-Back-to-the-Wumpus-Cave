# Assignment 2.0 (Warm-Up, Variant A): Find Back to the Wumpus Cave

This project implements an AI agent that navigates a grid-based environment to locate cave entrances ('W') under uncertain observations. The agent employs probabilistic reasoning to account for sensor inaccuracies and optimizes path planning to maximize success chances and minimize expected time using the A* algorithm with time constraints. It balances exploration efficiency with success probability while considering terrain-dependent movement costs.


## Table of Contents

- [Introduction](#introduction)
  - Key Features
- [Setup](#setup)
  - Repository content
  - How to run the code
  - Used libraries
- [Code Structure](#code-structure)
- [Self Evaluation and Design Decisions](#design-decision)
- [Output Format](#output-format)

## Introduction
This Python agent navigates uncertain terrain to locate cave entrances ('W') using probabilistic reasoning and adaptive pathfinding. <br>
**Core components include:**
- `astar`: for time-optimized search with terrain-aware movement costs.
- `compute_posterior`: for Bayesian position probability updates.
- `movement_cost`: modeling gear-dependent traversal penalties.
- `simulate_path_success`: validating paths against probabilistic start positions. 

The agent employs `find_best_plan` to evaluate paths using dual criteria: success probability (via posterior distributions) and time efficiency (through gear-adjusted terrain costs). Key innovations include misidentification handling (20% error rate for biome cells) and a Manhattan distance heuristic for cave proximity estimation. The main `agent_function` dynamically processes map observations, equipment status, and temporal constraints to generate action sequences. Through probabilistic simulations and priority-queue search, the system returns optimized plans with calculated success rates and expected completion times, adapting strategies to evolving environmental understanding.

### Key Features 
- **Uncertainty Handling**: Accounts for misidentified cell observations (20% error rate for 'B'/'C' cells).
- **Bayesian Posterior Updates:** Dynamic probability calculations for agent positioning.
- **Pathfinding with A\* Algorithm:** Optimizes routes using heuristic-based search.
- **Temporal Constraints:** Path validation against strict time limits.
- **Equipment Considerations**: Adjusts movement costs based on climbing gear availability.
  
## Setup
### This repository contains:
 1) **`example_agent.py`**: Core implementation of navigation logic
 2) **`client.py`**: A Python implementation of the AISysProj server protocol
 3) **agent-configs**: Folder contains the JSON file for the environments
 
### How to run the code: 
1) **`example_agent.py`**, **`client.py`** and **agent-configs** folder must all be on the same folder
2) Run the **cmd** on the current path.
3) Run the following command **python example_agent.py agent-configs/env-*.json** 

### Used libraries:
**_heapq:_**
mplements priority queues for efficient pathfinding in A* algorithm.
**_random:_**
Simulates sensor uncertainty in cell type observations (20% error rate).
**_logging:_**
Tracks runtime events and debugging information during environment interactions.

## Code Structure
1) **Library imports**
```python
from heapq import heappush, heappop
import random
```
- **heapq:** Used for implementing priority queues in the A* search algorithm.
- **random:** Simulates probabilistic sensor inaccuracies for cell type observations.

2) **Movement & Perception Functions**
```python
def move_agent(position, action):
    # ...
```
- Translates actions (e.g., "GO north") into position changes on the grid.
- Returns new coordinates based on movement direction
- Handles invalid actions by returning original position

```python
def perceived_cell_type(cell_type):
    # ...
```
- Simulates sensor inaccuracies with a 20% error rate for 'B' and 'C' cells.
- Perfect identification for other cell types
  
3) **Cost Calculation Function**
```python
def movement_cost(cell, climbing_gear=False):
    # ...
```
- Calculates traversal costs for different cell types, adjusting for climbing gear usage.
  
    |    Terrain    |   With Gear   |  Without Gear |
    | ------------- | ------------- | ------------- |
    |     M/B/C     |    1.2h       |      1.0h     |
    |       R       |    2.0h       |      4.0h     |
    |    Default    |    1.2h       |      1.0h     |

4) **Map Analysis Utilities:**
```python
def find_positions(map_lines, target):
    # ...
```
- Locates all positions of a target cell type (e.g., 'W') in the map.
- `find_positions(map, 'W')` finds all cave entrances

```python
def get_cell_type(map_lines, x, y):
    # ...
```
- Safe cell type lookup with boundary handling
- Treats out-of-bounds coordinates as meadows ('M')
- Valid coordinates return actual map character

5) **Probabilistic Reasoning Engine:**
```python
def compute_posterior(map_lines, observations):
    # ...
```
- Calculates position probabilities using Bayesian inference
- Handles two observation types: current-cell and cell-west
- Incorporates 20% sensor error rate for 'B'/'C' cells
- Uses uniform prior distribution across all cells
- **Workflow:**
   - Initialize uniform prior (1/total_cells)
   - Update likelihoods based on observations:
   - Normalize probabilities to sum to 1

6) **Pathfinding System:**
```python
def heuristic(position, map_lines):
    # ...
```
-  Estimates minimal time to reach nearest cave entrance
-  Manhattan distance to closest 'W' cell

```python
def astar(start, map_lines, max_time, climbing_gear, posterior, max_actions=5):
    # ...
```
- A* search implementation with time/action constraints
  
**Key Features:**
   - Priority queue using (time + heuristic) for ordering
   - Tracks visited (position, path) pairs to prevent loops
- **Enforces:**
   - Maximum 5 actions per path
   - Total time ≤ max_time
- **Cost Calculation:**
   - First move: 0.5h fixed cost
   - Subsequent moves: Terrain-dependent costs

7) **Plan Validation System:**
```python
def simulate_path_success(start, path, map_lines, max_time, climbing_gear):
    # ...
```
-  Validates if a path reaches 'W' within time limit
- **Time Accounting:**
   - Initial cell exit: 0.5h fixed cost
   - Movement costs applied after first action

8) **Optimization Engine:**
```python
def find_best_plan(map_lines, posterior, climbing_gear, max_time):
    # ...
```
- Evaluates all possible paths to find optimal solution
- **Evaluation Criteria:**
    - **Success Chance:** Σ(prob × path_success)
    - **Expected Time:** (Σ prob×time) / success_chance
    - **Rating:** success_chance×expected_time + (1-success_chance)×max_time
- **Workflow:**
    - Generate paths from all probable starting positions
    - Simulate each path's success probability
    - Select plan with minimal rating score

9) **Main Agent Function:**
```python
def agent_function(request_dict, _info):
    # ...
```    
- **Input Processing:**
   - Extracts map, equipment, time limit, and observations
   - Converts map string to 2D array
- **Core Logic:**
   - Compute posterior position probabilities
   - Check climbing gear availability
   - Find optimal plan using A* and probabilistic evaluation
- **Output:**
    ```python
      return {
          "actions": best_plan,
          "success-chance": success_chance,
          "expected-time": expected_time
      }
    ```

10) **Execution Flow:**
```python
if __name__ == '__main__':
    # ...
```
- **Runtime Configuration:**
    - Loads environment config from command line
    - Sets up logging system
- **Agent Interface:**
    - Integrates with provided `client.py`
    - Runs 100,000 simulations for performance evaluation

## Self Evaluation and Design Decisions

## Output Format

![Screenshot 2024-11-05 021618](https://github.com/user-attachments/assets/c87a469e-8126-4984-8621-f326a5f01972)












