
# Thompson Sampling Introduction
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random

# predefined win rates for each strategy (AI model is unaware of these)
conversion_rates = [0.05, 0.13, 0.09, 0.16, 0.11, 0.04, 0.20, 0.08, 0.01]
reward_values = [1, 1, 1, 1, 1, 1, 1, 1, 1]

N = 10000                   # number of customers
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
        random_beta = random.betavariate(number_of_rewards_1[i] + 1, number_of_rewards_0[i] + 1)
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
