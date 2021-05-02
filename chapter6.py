
# Thompson Sampling Introduction
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
from scipy.stats import beta

# predefined win rates for each strategy (AI model is unaware of these)
conversion_rates = [0.05, 0.13, 0.09, 0.16, 0.11, 0.04, 0.20, 0.08, 0.01]
reward_values = [1, 1, 1, 1, 1, 1, 1, 1, 1]

N = 5000                  # number of customers
d = len(conversion_rates)   # number of strategies

# run simulation to create data frame of slot machine wins / losses
X = np.array(np.zeros([N, d]))
machine_selected = []
for i in range(N):
    for j in range(d):
        if np.random.rand() < conversion_rates[j]:
            X[i][j] = 1

# models for determining the best strategy
# random selection
strategies_selected_rs = []
total_reward_rs = 0

# thompson sampling
strategies_selected_ts = []
total_reward_ts = 0
number_of_rewards_0 = [0] * d
number_of_rewards_1 = [0] * d

for n in range(N):
    # random selection logic
    strategy_rs = random.randrange(d)               # pick random strategy
    strategies_selected_rs.append(strategy_rs)      # log the selection
    reward_rs = X[n, strategy_rs]                   # was it a selected strategy in X? y = 1, n = 0
    total_reward_rs = total_reward_rs + reward_rs   # add the 1 or 0 to total_reward_rs

    # thompson sampling method
    strategy_ts = 0     # reset selected strategy
    max_random = 0      # max random for holding which is the highest selected strategy for that customer
    for i in range(d):
        # utilize beta function to analyze learning of each strategy
        random_beta = np.random.beta(number_of_rewards_1[i] + 1, number_of_rewards_0[i] + 1)
        # set the strategy with the highest beta
        if random_beta > max_random:
            max_random = random_beta
            strategy_ts = i

    reward_ts = X[n, strategy_ts]                   # was it a selected strategy in X? y = 1, n = 0

    if reward_ts == 1:
        number_of_rewards_1[strategy_ts] = number_of_rewards_1[strategy_ts] + 1     # set positive reward
    else:
        number_of_rewards_0[strategy_ts] = number_of_rewards_0[strategy_ts] + 1     # set negative reward

    strategies_selected_ts.append(strategy_ts)      # log the selection
    total_reward_ts = total_reward_ts + reward_ts   # add the 1 or 0 to total_reward_ts

# print Relative Return - how much better was TS than RS?
relative_return = (total_reward_ts - total_reward_rs) / total_reward_rs * 100
print("Relative Return: {:.0f} %".format(relative_return))

plt.hist(strategies_selected_ts)
plt.title('Histograms of Selections')
plt.xlabel('Strategy')
plt.ylabel('Number of times the strategy was selected')
plt.show()

# print("\nRewards By Strategy = ", number_of_rewards_1)
# print("\nNo Rewards By Strategy = ", number_of_rewards_0)

# rv0 = beta(number_of_rewards_1[0], number_of_rewards_0[0])
# rv1 = beta(number_of_rewards_1[1], number_of_rewards_0[1])
# rv2 = beta(number_of_rewards_1[2], number_of_rewards_0[2])
# rv3 = beta(number_of_rewards_1[3], number_of_rewards_0[3])
# rv4 = beta(number_of_rewards_1[4], number_of_rewards_0[4])
# rv5 = beta(number_of_rewards_1[5], number_of_rewards_0[5])
# rv6 = beta(number_of_rewards_1[6], number_of_rewards_0[6])
# rv7 = beta(number_of_rewards_1[7], number_of_rewards_0[7])
# rv8 = beta(number_of_rewards_1[8], number_of_rewards_0[8])

# x = np.linspace(0, .225, 200)
# plt.title('Beta Distribution By Slot Machine')
# plt.plot(x, rv0.pdf(x), label="Strategy 1 [" + str(round((number_of_rewards_1[0]/number_of_rewards_0[0])*100,1)) + "%]")
# plt.plot(x, rv1.pdf(x), label="Strategy 2 [" + str(round((number_of_rewards_1[1]/number_of_rewards_0[1])*100,1)) + "%]")
# plt.plot(x, rv2.pdf(x), label="Strategy 3 [" + str(round((number_of_rewards_1[2]/number_of_rewards_0[2])*100,1)) + "%]")
# plt.plot(x, rv3.pdf(x), label="Strategy 4 [" + str(round((number_of_rewards_1[3]/number_of_rewards_0[3])*100,1)) + "%]")
# plt.plot(x, rv4.pdf(x), label="Strategy 5 [" + str(round((number_of_rewards_1[4]/number_of_rewards_0[4])*100,1)) + "%]")
# plt.plot(x, rv5.pdf(x), label="Strategy 6 [" + str(round((number_of_rewards_1[5]/number_of_rewards_0[5])*100,1)) + "%]")
# plt.plot(x, rv6.pdf(x), label="Strategy 7 [" + str(round((number_of_rewards_1[6]/number_of_rewards_0[6])*100,1)) + "%]")
# plt.plot(x, rv7.pdf(x), label="Strategy 8 [" + str(round((number_of_rewards_1[7]/number_of_rewards_0[7])*100,1)) + "%]")
# plt.plot(x, rv8.pdf(x), label="Strategy 9 [" + str(round((number_of_rewards_1[8]/number_of_rewards_0[8])*100,1)) + "%]")
# plt.legend()
# plt.show()
