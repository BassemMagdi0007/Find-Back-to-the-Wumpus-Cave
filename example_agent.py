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

"""Move the agent to a new position based on the action."""
def move_agent(position, action):
    if action == "GO north":
        return (position[0] - 1, position[1])  # Move up
    elif action == "GO south":
        return (position[0] + 1, position[1])  # Move down
    elif action == "GO east":
        return (position[0], position[1] + 1)  # Move right
    elif action == "GO west":
        return (position[0], position[1] - 1)  # Move left
    else:
        return position  # No movement if action is unrecognized


"""Calculate the movement cost for a given cell type based on whether climbing gear is used."""
def movement_cost(cell, climbing_gear=False, is_starting_cell=False):
    # if is_starting_cell:
    #     return 0.5  # Cost to move from the center of the starting cell

    if cell in ["M", "B", "C"] and not is_starting_cell:
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


def bfs(start, game_map, climbing_gear):
    directions = ["GO north", "GO south", "GO east", "GO west"]
    
    queue = deque([(start, [])])  # Queue of (current position, path taken)
    visited = set()  # Track visited positions
    visited.add(start)

    while queue:
        current_position, path = queue.popleft()
        
        # Check if we reached the cave entrance
        if game_map[current_position[0]][current_position[1]] == 'W':
            return path  # Return the path to the cave entrance

        # Explore neighbors
        for action in directions:
            new_position = move_agent(current_position, action)
            if (0 <= new_position[0] < len(game_map) and
                0 <= new_position[1] < len(game_map[0]) and
                new_position not in visited):
                
                cell = game_map[new_position[0]][new_position[1]]
                if cell != 'X':  # Assuming 'X' is an obstacle
                    visited.add(new_position)
                    queue.append((new_position, path + [action]))

    return None  # Return None if no path is found


def agent_function(request_dict, _info):
    print(request_dict)
    # Fetch relevant data from the request_dict
    initial_equipment = request_dict.get('initial-equipment', [])
    game_map = request_dict.get('map', '')
    max_time = request_dict.get('max-time', 0.0)
    current_cell = request_dict['observations'].get('current-cell', '')

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

    # Check if the agent has climbing gear
    climbing_gear = 'climbing_gear' in initial_equipment

    all_actions = []  # To store actions from all start positions
    total_time = 0  # Total time for all paths found

        # Iterate through each start position to calculate time for the path to cave entrance
    for start in start_positions:
        # Perform BFS to find the path from start to cave entrance
        path = bfs(start, map_lines, climbing_gear)

        if path is not None:  # If a path is found
            actions = path  # Actions taken to reach the cave entrance
            total_time_for_path = 0.5  # Start with the initial move cost of 0.5 hours
            
            # Calculate total movement cost for the actions
            current_position = start  # Start from the initial position
            first_move = True  # Track the first move
            
            for action in actions:
                # Move the agent to the new position
                new_position = move_agent(current_position, action)

                # If it's not the first move, calculate movement cost based on cell type
                if not first_move:
                    cost = movement_cost(map_lines[new_position[0]][new_position[1]], climbing_gear)
                    total_time_for_path += cost  # Add cost to total time
                else:
                    # First move is already accounted as 0.5, set first_move to False
                    first_move = False

                # Update current position
                current_position = new_position  

            # Append actions and their respective time to the all_actions list
            all_actions.append(actions)
            total_time += total_time_for_path  # Add path time to total time
            
            # Print the actions and the total time required
            print(f"Actions for {start}: {actions}, Total Time: {total_time_for_path:.2f} hours")
        else:
            print(f"No path found from start position {start}.")

    # Dummy values for success chance and expected time
    success_chance = 0.8  # Dummy success chance
    expected_time = total_time  # Use the total time calculated

    return {
        "actions": all_actions, 
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