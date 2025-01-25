from collections import deque
import random
import heapq


"""Move the agent to a new position based on the action"""
# DONE
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


"""Calculate movement cost for a given cell type based on whether climbing gear is used"""
# DONE
def movement_cost(cell, climbing_gear):
    if cell in ["M", "B", "C"]:
        return 1.2 if climbing_gear else 1.0
    elif cell == "R":
        return 2.0 if climbing_gear else 4.0
    elif cell == "W":
        return 0.0  # Cave entrance
    return 1.2 if climbing_gear else 1.0  # Default to ground cell


"""Find positions of a target character in the map"""
# DONE
def find_positions(map_lines, target):
    return [(row, col) for row, line in enumerate(map_lines) for col, cell in enumerate(line) if cell == target]


"""Check the validity of a given position, within the map or treating out-of-bounds cells as meadows (M)"""
# DONE
def get_cell_type(map_lines, x, y):
    if 0 <= x < len(map_lines) and 0 <= y < len(map_lines[x]):
        return map_lines[x][y]
    return 'M'


"""Calculate the posterior probabilities for each cell based on given observations"""
# DONE
def compute_posterior(map_lines, observations):
    rows = len(map_lines)
    cols = len(map_lines[0]) if rows else 0
    total_cells = rows * cols
    prior = 1.0 / total_cells  # Uniform prior

    posterior = {}
    for x in range(rows):
        for y in range(cols):
            # Use get_cell_type to handle out-of-bounds positions
            cell_type = get_cell_type(map_lines, x, y)
            # cell_type = map_lines[x][y]
            likelihood = 1.0

            # Handle current-cell observation
            if 'current-cell' in observations:
                observed_type = observations['current-cell']
                if cell_type in ['B', 'C']:
                    if observed_type == cell_type:
                        likelihood *= 0.8  # Correct identification
                    elif (observed_type, cell_type) in [('B', 'C'), ('C', 'B')]:
                        likelihood *= 0.2  # Misidentification
                    else:
                        likelihood *= 0.0  # Impossible observation
                else:
                    likelihood *= 1.0 if observed_type == cell_type else 0.0

            posterior[(x, y)] = prior * likelihood

    # Normalize posterior probabilities
    total = sum(posterior.values())
    if total > 0:
        posterior = {k: v / total for k, v in posterior.items()}
    else:
        posterior = {k: 0.0 for k in posterior.keys()}  # No valid cells

    return posterior


"""Estimate remaining time to reach the goal (Manhattan distance * min movement cost)."""
def heuristic(position, goal):
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1]) * 1.0


"""Perform A* search to find the optimal path from the start position to a cave entrance (W) within the allowed time"""
def astar(start, map_lines, has_gear, max_time):
    directions = ["GO north", "GO south", "GO east", "GO west"]
    goals = find_positions(map_lines, 'W')  # Find all W cells

    if not goals:
        return None  # No cave entrance on the map

    # Priority queue: (f_score, cumulative_time, position, has_gear, path)
    queue = [(0, 0.5, start, has_gear, [])]
    visited = set([(start, has_gear)])  # Track visited (position, has_gear) states

    while queue:
        f_score, total_time, current_position, current_gear, path = heapq.heappop(queue)

        # Check if current cell is a cave entrance (W) on the original map
        if (0 <= current_position[0] < len(map_lines) and
            0 <= current_position[1] < len(map_lines[0]) and
            map_lines[current_position[0]][current_position[1]] == 'W'):
            return path

        # Explore neighbors (including out-of-bounds)
        for action in directions:
            new_position = move_agent(current_position, action)
            new_state = (new_position, current_gear)

            if new_state not in visited:
                # Calculate movement cost for the new cell
                cell_type = get_cell_type(map_lines, new_position[0], new_position[1])
                move_cost = movement_cost(cell_type, current_gear)
                # Check if this is the first action in the path
                if len(path) == 0:
                    # First action: time remains 0.5 (initial move)
                    new_total_time = total_time
                else:
                    new_total_time = total_time + move_cost

                if new_total_time <= max_time:
                    h_score = min(heuristic(new_position, goal) for goal in goals)
                    f_score = new_total_time + h_score

                    heapq.heappush(queue, (f_score, new_total_time, new_position, current_gear, path + [action]))
                    visited.add(new_state)

        # Add DROP climbing-gear action if the agent has gear
        if current_gear:
            new_state = (current_position, False)  # Drop gear
            if new_state not in visited:
                # No movement cost for dropping gear
                h_score = min(heuristic(current_position, goal) for goal in goals)
                f_score = total_time + h_score

                heapq.heappush(queue, (f_score, total_time, current_position, False, path + ["DROP climbing-gear"]))
                visited.add(new_state)

    return None  # No path to cave entrance found


"""Calculate the total time required for a given path based on cell types"""
def calculate_total_time(path, start, map_lines, has_gear):
    total_time = 0.5  # Initial move cost
    current_position = start
    current_gear = has_gear
    first_move = True

    for action in path:
        if action == "DROP climbing-gear":
            current_gear = False
            continue

        new_position = move_agent(current_position, action)
        if not first_move:
            cell_type = get_cell_type(map_lines, new_position[0], new_position[1])
            total_time += movement_cost(cell_type, current_gear)
        else:
            first_move = False  # Skip cost for the first move

        current_position = new_position

    return total_time


"""Simulate the movement along the path and check if it leads to a cave entrance."""
def simulate_path_success(start, path, map_lines, max_time, has_gear):
    current_position = start
    current_gear = has_gear
    total_time = 0.5  # Initial move cost of 0.5 hours
    first_move = True

    for action in path:
        if action == "DROP climbing-gear":
            current_gear = False
            continue

        new_position = move_agent(current_position, action)
        if not first_move:
            cell_type = get_cell_type(map_lines, new_position[0], new_position[1])
            total_time += movement_cost(cell_type, current_gear)
        else:
            first_move = False  # Skip movement cost for the first move

        current_position = new_position

    # Check if final position is a cave entrance (W)
    if (0 <= current_position[0] < len(map_lines) and
        0 <= current_position[1] < len(map_lines[0]) and
        map_lines[current_position[0]][current_position[1]] == 'W'):
        return total_time <= max_time
    return False


"""Find the best plan, prioritizing higher success chance and lower expected time"""
def find_best_plan(map_lines, posterior, has_gear, max_time):
    best_plan = None
    highest_success_chance = 0
    best_time = float('inf')
    total_time_for_successful_starts = 0.0

    # Get all possible start positions (all cells with non-zero posterior probability)
    start_positions = [pos for pos, prob in posterior.items() if prob > 0]

    for start in start_positions:
        path = astar(start, map_lines, has_gear, max_time)
        if path:
                # For each cell in posterior, check if path works and compute time from that cell
                success_chance = 0.0
                total_time_for_successful_starts = 0.0
                
                for cell, prob in posterior.items():
                    # Check if path is successful from this cell
                    is_success = simulate_path_success(cell, path, map_lines, max_time, has_gear)
                    if is_success:
                        # Calculate time starting from this cell
                        cell_time = calculate_total_time(path, cell, map_lines, has_gear)
                        success_chance += prob
                        total_time_for_successful_starts += prob * cell_time
                
                # Update best plan based on success_chance and total_time_for_successful_starts
                if success_chance > 0:
                    expected_time = total_time_for_successful_starts / success_chance
                else:
                    expected_time = 0.0
                
                # Compare and update best plan
                if success_chance > highest_success_chance or (success_chance == highest_success_chance and expected_time < best_time):
                    best_plan = path
                    highest_success_chance = success_chance
                    best_time = expected_time

    return best_plan, expected_time, highest_success_chance


"""Main agent function"""
def agent_function(request_dict, _info):
    print("\n")
    print('\t[Agent260]')
    print("request_dict:", request_dict)
    # Fetch data from the request dictionary
    initial_equipment = request_dict.get('initial-equipment', [])
    game_map = request_dict.get('map', '')
    max_time = request_dict.get('max-time', 0.0)
    observations = request_dict.get('observations', {})

    print('Initial Equipment:', initial_equipment)
    print('_________Game Map:_________\n', game_map)
    print('Max Time Allowed:', max_time)
    print('Current Cell:', observations.get('current-cell', ''))

    map_lines = [line.strip() for line in game_map.strip().split('\n')]

    # Compute posterior probabilities for all cells
    posterior = compute_posterior(map_lines, observations)

    # Check if the agent has climbing gear
    climbing_gear = 'climbing_gear' in initial_equipment

    # Find the best plan within the allowed time
    best_plan, expected_time, success_chance = find_best_plan(map_lines, posterior, climbing_gear, max_time)

    # Calculate the rating
    rating = success_chance * expected_time + (1 - success_chance) * max_time
    # Log for debugging
    print("BEST PLAN: ", best_plan)
    print("SUCCESS CHANCE: ", success_chance)
    print("EXPECTED TIME: ", expected_time)
    print("RATING: ", rating)
    print("_______________")

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
        run_limit=100000,     # stop after 1000 runs (then the rating is "complete")
        parallel_runs= False  # multiple requests are bundled in one server interaction (more efficient)
    )