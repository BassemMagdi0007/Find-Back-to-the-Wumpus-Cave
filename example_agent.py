"""
    To use this implementation, you simply have to implement `agent_function` such that it returns a legal action.
    You can then let your agent compete on the server by calling
        python3 example_agent.py path/to/your/config.json

    You can interrupt the script at any time and continue at a later time.
    The server will remember the actions you have sent.

    Note:
        By default the client bundles multiple requests for efficiency.
        This can complicate debugging.
        You can disable it by setting `parallel_runs=False` in the last line.
"""

# TODO
#... 

from collections import deque
import random

"""Move the agent to a new position based on the action"""
def move_agent(position, action):
    if action == "GO north":
        return (position[0] - 1, position[1])  # Move up
    elif action == "GO south":
        return (position[0] + 1, position[1])  # Move down
    elif action == "GO east":
        return (position[0], position[1] + 1)  # Move right
    elif action == "GO west":
        return (position[0], position[1] - 1)  # Move left
    return position  # No movement if action is unrecognized


def compute_posterior(map_lines, observed_cell_type):
    """Compute posterior probabilities for all cells given an observation."""
    likelihood = {}
    for row_idx, row in enumerate(map_lines):
        for col_idx, cell_type in enumerate(row):
            # Compute likelihood P(observation | actual cell type)
            if observed_cell_type in ['B', 'C']:
                if cell_type == observed_cell_type:
                    lik = 0.8  # Correct identification
                elif cell_type in ['B', 'C']:
                    lik = 0.2  # Misidentification between B/C
                else:
                    lik = 0.0  # Other cell types cannot produce this observation
            else:
                # No misidentification for M, R, W
                lik = 1.0 if cell_type == observed_cell_type else 0.0
            likelihood[(row_idx, col_idx)] = lik

    # Normalize to get posterior probabilities
    total_lik = sum(likelihood.values())
    if total_lik == 0:
        # Edge case: Uniform distribution if observation is impossible
        total_cells = len(map_lines) * len(map_lines[0])
        return {cell: 1.0 / total_cells for cell in likelihood}
    else:
        return {cell: lik / total_lik for cell, lik in likelihood.items()}



"""Calculate movement cost based on ACTUAL cell type (no misidentification)."""
def movement_cost(cell_type, climbing_gear=False):
    if cell_type == "M":
        return 1.2 if climbing_gear else 1.0
    elif cell_type in ["B", "C"]:  # Trees
        return 1.2 if climbing_gear else 1.0
    elif cell_type == "R":  # Rocks
        return 2.0 if climbing_gear else 4.0
    return 1.0  # Default (e.g., W or out-of-bounds M)


"""Find positions of a target character in the map"""
def find_positions(map_lines, target):
    return [(row, col) for row, line in enumerate(map_lines) for col, cell in enumerate(line) if cell == target]


"""Perform BFS to find the path from start to cave entrance"""
def bfs(start, map_lines):
    directions = ["GO north", "GO south", "GO east", "GO west"]
    queue = deque([(start, [])])  # Queue of (position, path taken)
    visited = set([start])  # Track visited positions

    while queue:
        current_position, path = queue.popleft()

        # Check if we reached the cave entrance
        if map_lines[current_position[0]][current_position[1]] == 'W':
            return path

        # Explore neighbors
        for action in directions:
            new_position = move_agent(current_position, action)
            if (0 <= new_position[0] < len(map_lines) and
                0 <= new_position[1] < len(map_lines[0]) and
                new_position not in visited):

                visited.add(new_position)
                queue.append((new_position, path + [action]))

    return None  # No path to cave entrance found


"""Calculate the total time required for a given path based on cell types"""
def calculate_total_time(path, start, map_lines, climbing_gear):
    total_time = 0.5  # Initial move cost
    current_position = start
    first_move = True

    for action in path:
        new_position = move_agent(current_position, action)
        if not first_move:
            cell = map_lines[new_position[0]][new_position[1]]
            total_time += movement_cost(cell, climbing_gear)  # Actual cell type
        else:
            first_move = False  # Skip additional cost for the first move

        current_position = new_position

    return total_time


"""Simulate the movement along the path and check if it leads to a cave entrance."""
def simulate_path_success(start, path, map_lines, max_time, climbing_gear):  # Add climbing_gear parameter
    current_position = start
    total_time = 0.0

    for action in path:
        new_position = move_agent(current_position, action)
        # Handle out-of-bounds as 'M'
        if not (0 <= new_position[0] < len(map_lines) and 0 <= new_position[1] < len(map_lines[0])):
            cell_type = 'M'
        else:
            cell_type = map_lines[new_position[0]][new_position[1]]
        total_time += movement_cost(cell_type, climbing_gear)  # Use gear status

        current_position = new_position

    last_cell = map_lines[current_position[0]][current_position[1]] if (0 <= current_position[0] < len(map_lines) and 0 <= current_position[1] < len(map_lines[0])) else 'M'
    return last_cell == 'W' and total_time <= max_time

"""Calculate the success chance based on starting positions."""
def calculate_success_chance_for_starts(start_positions, map_lines, path, max_time):
    success_count = 0
    total_starts = len(start_positions)

    for agent in start_positions:
        if simulate_path_success(agent, path, map_lines, max_time):
            success_count += 1

    # Calculate success chance
    if total_starts > 0:
        return success_count / total_starts  # Fraction of successful starts
    else:
        return 0.0  # No starts to evaluate

def evaluate_plan(plan, posterior, map_lines, max_time, climbing_gear):
    success_chance = 0.0
    total_time = 0.0
    for cell, prob in posterior.items():
        if simulate_path_success(cell, plan, map_lines, max_time, climbing_gear):  # Pass gear
            time_taken = calculate_total_time(plan, cell, map_lines, climbing_gear)
            success_chance += prob
            total_time += prob * time_taken
    expected_time = total_time / success_chance if success_chance > 0 else 0.0
    return success_chance, expected_time

"""Find the best plan, prioritizing higher success chance and lower expected time"""
def find_best_plan(posterior, map_lines, climbing_gear, max_time):
    best_rating = float('inf')
    best_plan = []
    best_success = 0.0
    best_time = 0.0

    # Generate plans from the most probable cells
    max_prob = max(posterior.values(), default=0)
    candidate_cells = [cell for cell, prob in posterior.items() if prob == max_prob]

    for cell in candidate_cells:
        path = bfs(cell, map_lines)
        if path:
            # Evaluate this plan
            success, exp_time = evaluate_plan(path, posterior, map_lines, max_time, climbing_gear)
            rating = success * exp_time + (1 - success) * max_time

            if rating < best_rating:
                best_plan = path
                best_rating = rating
                best_success = success
                best_time = exp_time

    return best_plan, best_time, best_success


"""Main agent function"""
def agent_function(request_dict, _info):
    print("\n")
    print('\t[Agent260]')
    # Fetch data from the request dictionary
    initial_equipment = request_dict.get('initial-equipment', [])
    game_map = request_dict.get('map', '')
    max_time = request_dict.get('max-time', 0.0)
    current_cell = request_dict['observations'].get('current-cell', '')

    map_lines = [line.strip() for line in game_map.strip().split('\n')]

    # Log the fetched information for debugging
    print('Initial Equipment:', initial_equipment)
    print('_________Game Map:_________\n', game_map)
    print('Max Time Allowed:', max_time)
    print('Current Cell:', current_cell)  # Directly print observed cell

    # Compute posterior probabilities
    observed_cell_type = current_cell
    posterior = compute_posterior(map_lines, observed_cell_type)
    
    # Check climbing gear
    climbing_gear = 'climbing-gear' in initial_equipment  # Ensure hyphen matches JSON key

    # Find best plan
    best_plan, expected_time, success_chance = find_best_plan(posterior, map_lines, climbing_gear, max_time)
    
    # Calculate rating
    rating = success_chance * expected_time + (1 - success_chance) * max_time
    print("BEST PLAN: ", best_plan)
    print("SUCCESS CHANCE: ", success_chance)
    print("EXPECTED TIME: ", expected_time)
    print("RATING: ", rating)

    return {
        "actions": best_plan if best_plan else [],
        "success-chance": success_chance,
        "expected-time": expected_time
    }
    

if __name__ == '__main__':
    try:
        from client import run
    except ImportError:
        raise ImportError('You need to have the client.py file in the same directory as this file')

    import logging
    logging.basicConfig(level=logging.INFO)

    import sys
    config_file = sys.argv[1]

    run(
        config_file,        # path to config file for the environment (in your personal repository)
        agent_function,
        processes=1,        # higher values will call the agent function on multiple requests in parallel
        run_limit=1000,     # stop after 1000 runs (then the rating is "complete")
        parallel_runs= False  # multiple requests are bundled in one server interaction (more efficient)
    )