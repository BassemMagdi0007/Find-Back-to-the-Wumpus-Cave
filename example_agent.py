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


"""Return the perceived cell type based on conditional probability."""
def perceived_cell_type(cell_type):
    if cell_type == 'B':
        return 'C' if random.random() < 0.2 else 'B'
    elif cell_type == 'C':
        return 'B' if random.random() < 0.2 else 'C'
    return cell_type  # No misidentification for other cell types


"""Calculate movement cost for a given cell type based on whether climbing gear is used"""
def movement_cost(cell, climbing_gear=False):
    perceived_type = perceived_cell_type(cell)
    if perceived_type in ["M", "B", "C"]:
        return 1.2 if climbing_gear else 1.0
    elif perceived_type == "R":
        return 2.0 if climbing_gear else 4.0
    return 1.2 if climbing_gear else 1.0  # Default to ground cell


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
            perceived_type = perceived_cell_type(cell)
            total_time += movement_cost(perceived_type, climbing_gear)
        else:
            first_move = False  # Skip additional cost for the first move

        current_position = new_position

    return total_time


"""Simulate the movement along the path and check if it leads to a cave entrance."""
def simulate_path_success(start, path, map_lines, max_time):
    current_position = start
    total_time = 0.0  # Initialize total time for the path

    for action in path:
        # Move the agent
        new_position = move_agent(current_position, action)

        # FIXME will tackle the code in the maps dealing with outside Meadows
        # Check if the new position is within bounds before accessing map_lines
        if not (0 <= new_position[0] < len(map_lines) and 0 <= new_position[1] < len(map_lines[0])):
            return False  # Out of bounds, so return failure

        # Calculate the movement cost for the new position
        cell_type = map_lines[new_position[0]][new_position[1]]
        perceived_type = perceived_cell_type(cell_type)
        total_time += movement_cost(perceived_type, climbing_gear=True)  # Assuming climbing gear for simplicity

        # Update the current position
        current_position = new_position

    # Check if the last position is the cave entrance and total time is within limits
    last_cell_type = map_lines[current_position[0]][current_position[1]]
    return last_cell_type == 'W' and total_time <= max_time  # Return True if successful


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
    

"""Return the start cell type after applying a 20% misidentification chance."""
def apply_start_cell_misidentification(cell_type):
    if cell_type == 'B' and random.random() < 0.2:
        return 'C'  # 20% chance to misidentify 'B' as 'C'
    elif cell_type == 'C' and random.random() < 0.2:
        return 'B'  # 20% chance to misidentify 'C' as 'B'
    return cell_type  # No misidentification


"""Find the best plan, prioritizing higher success chance and lower expected time"""
def find_best_plan(start_positions, cave_entrances, map_lines, climbing_gear, max_time):
    best_plan = None
    highest_success_chance = 0
    best_time = float('inf')
    total_time_for_successful_starts = 0.0
    successful_start_count = 0

    for start in start_positions:
        path = bfs(start, map_lines)
        if path:
            total_time_for_path = calculate_total_time(path, start, map_lines, climbing_gear)
            if total_time_for_path <= max_time:
                # Calculate success chance for this path from start positions
                success_chance = calculate_success_chance_for_starts(start_positions, map_lines, path, max_time)
                
                # Update expected time calculations only if this path is successful
                for agent in start_positions:
                    # Check if this specific start position would lead to a successful path
                    if simulate_path_success(agent, path, map_lines, max_time):
                        total_time_for_successful_starts += total_time_for_path
                        successful_start_count += 1

                # Choose the best plan based on success chance and time
                if success_chance > highest_success_chance or (success_chance == highest_success_chance and total_time_for_path < best_time):
                    best_plan = path
                    highest_success_chance = success_chance
                    best_time = total_time_for_path

    # Calculate the expected time for the successful starts only
    if successful_start_count > 0:
        expected_time = total_time_for_successful_starts / successful_start_count
    else:
        expected_time = 0.0

    return best_plan, expected_time, highest_success_chance


"""Main agent function"""
def agent_function(request_dict, _info):
    print("\n")
    print("request_dict:", request_dict)
    # Fetch data from the request dictionary
    initial_equipment = request_dict.get('initial-equipment', [])
    game_map = request_dict.get('map', '')
    max_time = request_dict.get('max-time', 0.0)
    current_cell = request_dict['observations'].get('current-cell', '')

    map_lines = [line.strip() for line in game_map.strip().split('\n')]

    # Apply misidentification chance to the start cell
    start_cell = apply_start_cell_misidentification(current_cell)

    # Only print the message for cells 'B' and 'C'
    if current_cell in ['B', 'C']:
        print(f"Original Start Cell: {current_cell}, After Misidentification: {start_cell}")

    # Log the fetched information for debugging
    print('Initial Equipment:', initial_equipment)
    print('_________Game Map:_________\n', game_map)
    print('Max Time Allowed:', max_time)
    print('Current Cell:', start_cell)

    # Find cave entrances and start positions using the misidentified start_cell
    cave_entrances = find_positions(map_lines, 'W')
    start_positions = find_positions(map_lines, start_cell)  # Use start_cell instead of current_cell

    # Check if the agent has climbing gear
    climbing_gear = 'climbing_gear' in initial_equipment

    # Find the best plan within the allowed time
    best_plan, expected_time, success_chance = find_best_plan(start_positions, cave_entrances, map_lines, climbing_gear, max_time)

    # Calculate the rating
    rating = success_chance * expected_time + (1 - success_chance) * max_time
    # Log for debugging
    print("BEST PLAN: ", best_plan)
    print("SUCCESS CHANCE: ", success_chance)
    print("EXPECTED TIME: ", expected_time)
    print("RATING: ", rating)

    # Return the best plan if found, otherwise return empty result
    if best_plan:
        return {
            "actions": best_plan,  
            "success-chance": success_chance,
            "expected-time": expected_time
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