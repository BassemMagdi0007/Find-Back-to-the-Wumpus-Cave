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
from collections import deque

# Calculate movement cost based on cell type
# Cell takes 1 hour (M,B,C) without climbing gear
# With climbing gear it takes 1.2 hour
# Cell with rocks (R) takes 4 hours without climbing gear
# with climbing gear it takes 2 hours

"""Calculate the movement cost for a given cell type based on whether climbing gear is used."""
def movement_cost(cell, climbing_gear=False):
    if cell in ["M", "B", "C"]:
        return 1.2 if climbing_gear else 1.0
    elif cell == "R":
        return 2.0 if climbing_gear else 4.0
    else:
        # Treat unknown cells as 'M' by default
        return 1.2 if climbing_gear else 1.0
    
"""Find positions of a target character in the map."""   
def find_positions(map_lines, target):
    positions = []
    for row, line in enumerate(map_lines):
        for col, cell in enumerate(line):
            if cell == target:
                positions.append((row, col))
    return positions

from collections import deque

def bfs_find_path(start, goals, map_lines, has_climbing_gear, max_time):
    """Find the shortest path from start to one of the goals using BFS."""
    queue = deque([(start, [])])  # Queue of (current_position, path)
    visited = set()

    while queue:
        (row, col), path = queue.popleft()
        if (row, col) in goals:
            return path  # Return the shortest path found to a cave

        if (row, col) in visited:
            continue
        visited.add((row, col))

        # Explore neighbors in N, E, S, W directions
        for dr, dc, direction in [(-1, 0, "GO north"), (1, 0, "GO south"),
                                  (0, 1, "GO east"), (0, -1, "GO west")]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < len(map_lines) and 0 <= new_col < len(map_lines[0]):
                cell_type = map_lines[new_row][new_col]
                cost = movement_cost(cell_type, has_climbing_gear)

                # Add the neighbor to the queue if within max_time
                if cost <= max_time:
                    queue.append(((new_row, new_col), path + [direction]))

    return []  # No path found within constraints

def agent_function(request_dict, _info):
    print(request_dict)
    # Fetch relevant data from the request_dict
    initial_equipment = request_dict.get('initial-equipment', [])
    game_map = request_dict.get('map', '')
    max_time = request_dict.get('max-time', 0.0)
    current_cell = request_dict['observations'].get('current-cell', '')
    has_climbing_gear = 'climbing-gear' in initial_equipment

    # map_lines = game_map.strip().split('\n')
    map_lines = [line.strip() for line in game_map.strip().split('\n')]

    # Log the fetched information for debugging
    print('Initial Equipment:', initial_equipment)
    print('Game Map:\n', game_map)
    print('Max Time Allowed:', max_time)
    print('Current Cell:', current_cell)

    # Find cave and starting position
    cave_entrances = find_positions(map_lines, 'W')
    start_positions = find_positions(map_lines, current_cell)

    # Log for debugging
    print('Cave Entrances:', cave_entrances)
    print('Start Positions:', start_positions)

    # Select a start position
    if not start_positions:
        return {"actions": [], "success-chance": 0.0, "expected-time": max_time}
    
    start = start_positions[0]  # Using the first valid start position for this example
    actions = bfs_find_path(start, cave_entrances, map_lines, has_climbing_gear, max_time)

    # Calculate the estimated time based on the movement cost along the path
    estimated_time = 0
    agent = start
    for action in actions:
        # Update position based on action
        if action == "GO north":
            agent = (agent[0] - 1, agent[1])
        elif action == "GO south":
            agent = (agent[0] + 1, agent[1])
        elif action == "GO east":
            agent = (agent[0], agent[1] + 1)
        elif action == "GO west":
            agent = (agent[0], agent[1] - 1)


    # Accumulate movement cost
    cell_type = map_lines[agent[0]][agent[1]]
    estimated_time += movement_cost(cell_type, has_climbing_gear)

    # Ensure estimated time does not exceed max time allowed
    estimated_time = min(estimated_time, max_time)

    # Estimate success chance
    success_chance = len(start_positions) / (len(map_lines) * len(map_lines[0]))

    print("Agent Response:")
    print("Actions:", actions)
    print("Success Chance:", success_chance)
    print("Expected Time:", estimated_time)

    return {
        "actions": actions,
        "success-chance": success_chance,
        "expected-time": estimated_time
    }

    # return {"actions": ["GO south", "GO east"], "success-chance": 0.5, "expected-time": 1.5}
    # return action


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
        parallel_runs=False  # multiple requests are bundled in one server interaction (more efficient)
    )
