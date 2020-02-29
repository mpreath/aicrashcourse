
import numpy as np

# predefined win rates for each slot machine (AI model is unaware of these)
conversion_rates = [0.15, 0.04, 0.13, 0.11, 0.10]

N = 10000                   # number of samples
d = len(conversion_rates)   # number of slot machines

# run simulation to create data frame of slot machine wins / losses
slot_machine_data = np.zeros((N, d))
for i in range(N):
    for j in range(d):
        if np.random.rand() < conversion_rates[j]:
            slot_machine_data[i][j] = 1

# model for determining the best slot machine
positive_reward = np.zeros(d)
negative_reward = np.zeros(d)

for i in range(N):
    selected = 0
    max_random = 0
    for j in range(d):
        # distribution graph for the slot machines will shift to the right for the best slot machine
        random_guess = np.random.beta(positive_reward[j] + 1, negative_reward[j] + 1)
        if random_guess > max_random:
            max_random = random_guess
            selected = j   

    # distribute positive reward if the selected slot machine won, negative if not.  
    if slot_machine_data[i][selected] == 1:
        positive_reward[selected] += 1
    else:
        negative_reward[selected] += 1

# display the details of the results
print(positive_reward)
print(negative_reward)
selected_machines = positive_reward + negative_reward
for i in range(d):
    print('Machine number ' + str(i+1) + ' was selected ' + str(selected_machines[i]) + ' times')
print('Conclusion: Best machine is machine number ' + str(np.argmax(selected_machines) + 1))
    
