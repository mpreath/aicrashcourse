
# Thompson Sampling Introduction
from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt

# predefined win rates for each slot machine (AI model is unaware of these)
conversion_rates = [0.10, 0.14, 0.05, 0.08, 0.12]
reward_values = [1, 1, 1, 1, 1]

N = 5000                   # number of samples
d = len(conversion_rates)   # number of slot machines

# run simulation to create data frame of slot machine wins / losses
slot_machine_data = np.zeros((N, d))
machine_selected = []
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
        # print(str(random_guess) + " : " + str(max_random))
        if random_guess > max_random:
            max_random = random_guess
            selected = j   

    machine_selected.append(selected)
    # distribute positive reward if the selected slot machine won, negative if not.  
    if slot_machine_data[i][selected] == 1:
        positive_reward[selected] += reward_values[selected]
    else:
        negative_reward[selected] += 1

# display the details of the results
# print(positive_reward)
# print(negative_reward)
# selected_machines = positive_reward + negative_reward
# for i in range(d):
#     print('Machine number ' + str(i+1) + ' was selected ' + str(selected_machines[i]) + ' times')
# print('Conclusion: Best machine is machine number ' + str(np.argmax(selected_machines) + 1))
print("\nRewards By Machine = ", positive_reward)
print("\nNo Rewards By Machine = ", negative_reward)
# print("\nTotal Rewards = ")
# print("\nMachine Selected At Each Round : ", machine_selected)

# plt.bar(['B1','B2','B3','B4','B5'],positive_reward)
# plt.title('MABP')
# plt.xlabel('Bandits')
# plt.ylabel('Reward By Each Machine')
# plt.show()

# from collections import Counter
# print("\nNumber of Times Each Machine Was Selected: ", dict(Counter(machine_selected)))
# print("\n")

# plt.hist(machine_selected)
# plt.title('Histogram of machines selected')
# plt.xlabel('Bandits')
# plt.xticks(range(0, 5))
# plt.ylabel('No. Of Times Each Bandit Was Selected')
# plt.show()

rv0 = beta(positive_reward[0], negative_reward[0])
rv1 = beta(positive_reward[1], negative_reward[1])
rv2 = beta(positive_reward[2], negative_reward[2])
rv3 = beta(positive_reward[3], negative_reward[3])
rv4 = beta(positive_reward[4], negative_reward[4])

x = np.linspace(0, .2, 200)
plt.title('Beta Distribution By Slot Machine')
plt.plot(x, rv0.pdf(x), label="Machine A [" + str(round((positive_reward[0]/negative_reward[0])*100,1)) + "%]")
plt.plot(x, rv1.pdf(x), label="Machine B [" + str(round((positive_reward[1]/negative_reward[1])*100,1)) + "%]")
plt.plot(x, rv2.pdf(x), label="Machine C [" + str(round((positive_reward[2]/negative_reward[2])*100,1)) + "%]")
plt.plot(x, rv3.pdf(x), label="Machine D [" + str(round((positive_reward[3]/negative_reward[3])*100,1)) + "%]")
plt.plot(x, rv4.pdf(x), label="Machine E [" + str(round((positive_reward[4]/negative_reward[4])*100,1)) + "%]")
plt.legend()
plt.show()
