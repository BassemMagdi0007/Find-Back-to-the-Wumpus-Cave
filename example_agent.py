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
# 1) Calculate the movement cost dynamically inside the BFS instead of agent_function
# 2) Instead of returning all the plans in all_actions. 
# -> Check for each plan if it was successful in finding the cave entrance (Y: Check next condition, N: Drop it)
# -> CHeck if the total time of that plan exceeded the max-time (Y: Drop it, N: Save it)
# FIXME
# How to calculate the success chance "the Wumpus is in a cell, for which the plan is successful and the plan succeeds for two of them" ?


from collections import deque
import random

# Move the agent to a new position based on the action
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


# Helper function for the agent's perception based on conditional probability
def perceived_cell_type(cell_type):
    """Return the perceived cell type based on conditional probability."""
    if cell_type == 'B':
        return 'C' if random.random() < 0.2 else 'B'
    elif cell_type == 'C':
        return 'B' if random.random() < 0.2 else 'C'
    return cell_type  # No misidentification for other cell types


# Calculate movement cost for a given cell type based on whether climbing gear is used
def movement_cost(cell, climbing_gear=False):
    perceived_type = perceived_cell_type(cell)
    if perceived_type in ["M", "B", "C"]:
        return 1.2 if climbing_gear else 1.0
    elif perceived_type == "R":
        return 2.0 if climbing_gear else 4.0
    return 1.2 if climbing_gear else 1.0  # Default to ground cell


# Find positions of a target character in the map
def find_positions(map_lines, target):
    return [(row, col) for row, line in enumerate(map_lines) for col, cell in enumerate(line) if cell == target]


# Perform BFS to find the path from start to cave entrance
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
                
                cell = map_lines[new_position[0]][new_position[1]]
                if cell != 'X':  # 'X' represents an obstacle
                    visited.add(new_position)
                    queue.append((new_position, path + [action]))

    return None  # No path to cave entrance found


# Calculate the total time required for a given path based on cell types
def calculate_total_time(path, start, map_lines, climbing_gear):
    total_time = 0.5  # Initial move cost
    current_position = start
    first_move = True

    for action in path:
        new_position = move_agent(current_position, action)
        if not first_move:
            cell = map_lines[new_position[0]][new_position[1]]
            perceived_type = perceived_cell_type(cell)
            total_time += movement_cost(perceived_type, climbing_gear)
        else:
            first_move = False  # Skip additional cost for the first move

        current_position = new_position

    return total_time


# Find the best plan from all start positions that meets time and destination criteria
def find_best_plan(start_positions, cave_entrances, map_lines, climbing_gear, max_time):
    best_plan = None
    min_total_time = float('inf')

    # Iterate through each start position
    for start in start_positions:
        path = bfs(start, map_lines)

        if path:  # If a path is found
            total_time_for_path = calculate_total_time(path, start, map_lines, climbing_gear)

            # Check if the path meets the criteria
            if total_time_for_path <= max_time and total_time_for_path < min_total_time:
                best_plan = path
                min_total_time = total_time_for_path

    return best_plan, min_total_time


# Main agent function
def agent_function(request_dict, _info):
    print("\n")
    print("request_dict:", request_dict)
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
    print('Current Cell:', current_cell)

    # Find cave entrances and start positions
    cave_entrances = find_positions(map_lines, 'W')
    start_positions = find_positions(map_lines, current_cell)

    # Log for debugging
    print('Cave Entrances:', cave_entrances)
    print('Start Positions:', start_positions)

    # Check if the agent has climbing gear
    climbing_gear = 'climbing_gear' in initial_equipment

    # Find the best plan within the allowed time
    best_plan, min_total_time = find_best_plan(start_positions, cave_entrances, map_lines, climbing_gear, max_time)

    
    # Log for debugging
    print("BEST PLAN: ", best_plan)

    # Return the best plan if found, otherwise return empty result
    if best_plan:
        return {
            "actions": best_plan,
            "success-chance": 0.8,  # Placeholder success chance
            "expected-time": min_total_time
        }
    else:
        return {
            "actions": [],
            "success-chance": 0.0,
            "expected-time": 0.0
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