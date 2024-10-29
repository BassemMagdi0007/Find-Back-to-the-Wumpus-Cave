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
    
"""Find positions of a target character in the map."""   
def find_positions(map_lines, target):
    positions = []
    for row, line in enumerate(map_lines):
        for col, cell in enumerate(line):
            if cell == target:
                positions.append((row, col))
    return positions


def agent_function(request_dict, _info):
    print(request_dict)
    # Fetch relevant data from the request_dict
    initial_equipment = request_dict.get('initial-equipment', [])
    game_map = request_dict.get('map', '')
    max_time = request_dict.get('max-time', 0.0)
    current_cell = request_dict['observations'].get('current-cell', '')

    # map_lines = game_map.strip().split('\n')
    map_lines = [line.strip() for line in game_map.strip().split('\n')]
    map_lines.reverse()

    # Log the fetched information for debugging
    print('Initial Equipment:', initial_equipment)
    # print('Game Map:\n', game_map)\
    print('Game Map:')
    for line in map_lines:
        print(line)
    print('Max Time Allowed:', max_time)
    print('Current Cell:', current_cell)

    # Find cave and starting position
    cave_entrances = find_positions(map_lines, 'W')
    start_positions = find_positions(map_lines, current_cell)

    # Log for debugging
    print('Cave Entrances:', cave_entrances)
    print('Start Positions:', start_positions)

    # Placeholder: Implement pathfinding here
    actions = ["GO south", "GO east"]  # Replace with calculated actions

    # Calculate estimated time based on movement costs and paths (using movement_cost)
    expected_time = 0  # Placeholder, replace with actual path time calculation

    # Estimate success chance based on the starting positions
    success_chance = 1.0 if start_positions else 0.0  # Simplified; adjust for probability

    # return {
    #     "actions": actions,
    #     "success-chance": success_chance,
    #     "expected-time": expected_time
    # }

    return {"actions": ["GO south", "GO east"], "success-chance": 0.5, "expected-time": 1.5}
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
