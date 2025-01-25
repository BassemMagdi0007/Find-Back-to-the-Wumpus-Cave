import random
from heapq import heappush, heappop

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
    if cell in ["M", "B", "C"]:
        return 1.2 if climbing_gear else 1.0
    elif cell == "R":
        return 2.0 if climbing_gear else 4.0
    return 1.2 if climbing_gear else 1.0  # Default to ground cell


"""Find positions of a target character in the map"""
def find_positions(map_lines, target):
    return [(row, col) for row, line in enumerate(map_lines) for col, cell in enumerate(line) if cell == target]


"""Return cell type, treating out-of-bounds cells as meadows (M)"""
def get_cell_type(map_lines, x, y):
    if 0 <= x < len(map_lines) and 0 <= y < len(map_lines[x]):
        return map_lines[x][y]
    return 'M'

def compute_posterior(map_lines, observations):
    rows = len(map_lines)
    cols = len(map_lines[0]) if rows else 0
    total_cells = rows * cols
    prior = 1.0 / total_cells  # Uniform prior

    posterior = {}
    for x in range(rows):
        for y in range(cols):
            cell_type = map_lines[x][y]
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

            # Handle cell-west observation (if present)
            if 'cell-west' in observations:
                west_x, west_y = x, y - 1
                if 0 <= west_y < cols:
                    west_type = map_lines[west_x][west_y]
                    observed_west = observations['cell-west']
                    if west_type in ['B', 'C']:
                        if observed_west == west_type:
                            likelihood *= 0.8
                        elif (observed_west, west_type) in [('B', 'C'), ('C', 'B')]:
                            likelihood *= 0.2
                        else:
                            likelihood *= 0.0
                    else:
                        likelihood *= 1.0 if observed_west == west_type else 0.0

            posterior[(x, y)] = prior * likelihood

    # Normalize posterior probabilities
    total = sum(posterior.values())
    if total > 0:
        posterior = {k: v / total for k, v in posterior.items()}
    else:
        posterior = {k: 0.0 for k in posterior.keys()}  # No valid cells

    return posterior

"""Estimate the minimum time to reach the nearest W cell, considering cell types."""
def heuristic(position, map_lines):
    min_time = float('inf')
    rows, cols = len(map_lines), len(map_lines[0])

    # Iterate over all cells to find the nearest W
    for x in range(rows):
        for y in range(cols):
            if map_lines[x][y] == 'W':
                # Calculate Manhattan distance
                distance = abs(position[0] - x) + abs(position[1] - y)

                # Estimate traversal time based on cell types along the path
                # Assume the path is a straight line (optimistic estimate)
                time = 0.5  # Initial 0.5-hour cost to exit the starting cell
                current_x, current_y = position[0], position[1]

                # Simulate the path to the W cell
                for _ in range(distance):
                    # Move towards the goal (x, y)
                    if current_x < x:
                        current_x += 1
                    elif current_x > x:
                        current_x -= 1
                    elif current_y < y:
                        current_y += 1
                    elif current_y > y:
                        current_y -= 1

                    # Get cell type and add traversal time
                    cell = get_cell_type(map_lines, current_x, current_y)
                    time += movement_cost(cell)

                # Update minimum time
                if time < min_time:
                    min_time = time

    return min_time


def astar(start, map_lines, max_time, climbing_gear, posterior, max_actions=10):
    directions = ["GO north", "GO south", "GO east", "GO west"]
    heap = [(0.5 + heuristic(start, map_lines), 0.5, start, [])]  # (priority, time, position, path)
    visited = set([(start, tuple())])  # Track position and path to avoid loops

    while heap:
        priority, current_time, current_position, path = heappop(heap)

        # Skip paths exceeding action limit
        if len(path) >= max_actions:
            continue

        # Check if current cell is W and time is within limit
        if (0 <= current_position[0] < len(map_lines) and
            0 <= current_position[1] < len(map_lines[0]) and
            map_lines[current_position[0]][current_position[1]] == 'W'):
            if current_time <= max_time:
                return path  # Return the path if valid

        # Prune paths exceeding max_time
        if current_time > max_time:
            continue

        # Explore neighbors
        for action in directions:
            new_position = move_agent(current_position, action)
            new_path = path + [action]

            # Calculate time for this action
            if len(new_path) == 1:
                new_time = current_time  # First action cost already included
            else:
                cell = get_cell_type(map_lines, new_position[0], new_position[1])
                cost = movement_cost(cell)
                new_time = current_time + cost

            # Calculate priority (f = g + h)
            priority = new_time + heuristic(new_position, map_lines)

            # Add to heap if not visited
            if (new_position, tuple(new_path)) not in visited:
                visited.add((new_position, tuple(new_path)))
                heappush(heap, (priority, new_time, new_position, new_path))

    return None


"""Simulate the movement along the path and check if it leads to a cave entrance."""
def simulate_path_success(start, path, map_lines, max_time, climbing_gear):
    current_position = start
    total_time = 0.5  # Initial 0.5-hour cost to exit the starting cell
    first_move = True

    # Check if starting cell is W (rare, but possible if misidentified)
    if (0 <= current_position[0] < len(map_lines) and
        0 <= current_position[1] < len(map_lines[0]) and
        map_lines[current_position[0]][current_position[1]] == 'W'):
        return (total_time <= max_time), total_time

    for action in path:
        new_position = move_agent(current_position, action)
        if not first_move:
            cell_type = get_cell_type(map_lines, new_position[0], new_position[1])
            total_time += movement_cost(cell_type, climbing_gear)
        else:
            first_move = False  # Skip adding cost for the first move
        current_position = new_position

        # Check after each move if current cell is W and time is within limit
        if (0 <= current_position[0] < len(map_lines) and
            0 <= current_position[1] < len(map_lines[0]) and
            map_lines[current_position[0]][current_position[1]] == 'W'):
            return (total_time <= max_time), total_time

    return False, 0.0  # Return failure if no W found


def find_best_plan(map_lines, posterior, climbing_gear, max_time):
    best_plan = None
    best_rating = float('inf')
    best_success_chance = 0
    best_expected_time = 0

    for start in [pos for pos, prob in posterior.items() if prob > 0]:
        path = astar(start, map_lines, max_time, climbing_gear, posterior, max_actions=5)
        if not path:
            continue

        # Calculate success chance and expected time for this path
        success_chance = 0.0
        total_weighted_time = 0.0

        for cell, prob in posterior.items():
            succeeds, time = simulate_path_success(cell, path, map_lines, max_time, climbing_gear)
            if succeeds:
                success_chance += prob
                total_weighted_time += prob * time

        # Calculate expected time (only if success_chance > 0)
        expected_time = total_weighted_time / success_chance if success_chance > 0 else 0.0

        # Calculate rating
        rating = success_chance * expected_time + (1 - success_chance) * max_time

        # Update best plan if this one has a lower rating
        if rating < best_rating:
            best_plan = path
            best_rating = rating
            best_success_chance = success_chance
            best_expected_time = expected_time

    return best_plan, best_expected_time, best_success_chance


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