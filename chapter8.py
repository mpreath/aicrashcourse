import numpy as np


def route(starting_location, ending_location):
    best_route = [starting_location]
    next_location = starting_location
    while next_location != ending_location:
        starting_state = location_to_state[starting_location]
        route_next_state = np.argmax(Q[starting_state, ])
        next_location = state_to_location[route_next_state]
        best_route.append(next_location)
        starting_location = next_location
    return best_route


# Set parameters for the Q-Learning
gamma = 0.75
alpha = 0.9

# map location letters to state integers
location_to_state = {'A': 0,
                     'B': 1,
                     'C': 2,
                     'D': 3,
                     'E': 4,
                     'F': 5,
                     'G': 6,
                     'H': 7,
                     'I': 8,
                     'J': 9,
                     'K': 10,
                     'L': 11}

# map state integers to location letters
state_to_location = {state: location for location, state in location_to_state.items()}

# list of possible actions
actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# rewards array, G is set to a high value since that is our goal
# each 1 represents a valid direction in the maze
R = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0, 0, 1000, 1, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]])

# initialize Q-values to 0
Q = np.array(np.zeros([12, 12]))

# train until TD for Q-values is converged
for i in range(1000):
    current_state = np.random.randint(0, 12)
    playable_actions = []
    for j in range(12):
        if R[current_state, j] > 0:
            playable_actions.append(j)
    next_state = np.random.choice(playable_actions)
    TD = R[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
    Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD

print("Q-values:")
print(Q.astype(int))

# output the best route
print('Route:')
print(route('E', 'G'))