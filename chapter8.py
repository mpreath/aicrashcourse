import numpy as np


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

# rewards array
#
#              A  B  C  D  E  F  G  H  I  J  K  L
R = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],             # A
              [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],             # B
              [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],             # C
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],             # D
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],             # E
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],             # F
              [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],             # G
              [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],             # H
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],             # I
              [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],             # J
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],             # K
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]])            # L


def route(starting_location, ending_location):
    R_new = np.copy(R)
    ending_state = location_to_state[ending_location]
    R_new[ending_state, ending_state] = 1000
    # initialize Q-values to 0
    Q = np.array(np.zeros([12, 12]))
    # train until TD for Q-values is converged
    for i in range(1000):
        current_state = np.random.randint(0, 12)
        playable_actions = []
        for j in range(12):
            if R_new[current_state, j] > 0:
                playable_actions.append(j)
        next_state = np.random.choice(playable_actions)
        TD = R_new[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[
            current_state, next_state]
        Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD

    calculated_route = [starting_location]
    next_location = starting_location
    while next_location != ending_location:
        starting_state = location_to_state[starting_location]
        route_next_state = np.argmax(Q[starting_state, ])
        next_location = state_to_location[route_next_state]
        calculated_route.append(next_location)
        starting_location = next_location
    return calculated_route


def best_route(starting_location, intermediate_location, ending_location):
    return route(starting_location, intermediate_location) + route(intermediate_location, ending_location)[1:]


# output the best route
print('\nRoute: ', best_route('E', 'K', 'G'))
