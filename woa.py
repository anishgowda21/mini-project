
import numpy as np
import math
import random
import os
import matplotlib.pyplot as plt
from test_functions import *
from numpy import abs


def whale_initial_position(hunting_party=5, min_values=[-5, -5], max_values=[5, 5], target_function=target_function):
    positions = np.zeros((hunting_party, len(min_values)+1))
    for i in range(hunting_party):
        for j in range(len(min_values)):
            positions[i][j] = random.uniform(min_values[j], max_values[j])
        positions[i][len(min_values)] = target_function(
            positions[i][:len(min_values)])
    return positions


def whale_leader_position(dimension=2, target_function=target_function):
    leader = np.zeros((1, dimension+1))
    for j in range(dimension):
        leader[0, j] = 0.0
    leader[0, -1] = target_function(leader[0, :dimension])
    return leader


def whale_update_leader(position, leader):
    for i in range(position.shape[0]):
        if (leader[0, -1] > position[i, -1]):
            for j in range(position.shape[1]):
                leader[0, j] = position[i, j]
    return leader


def whale_update_position(position, leader, a_linear_component=2, b_linear_component=1, spiral_param=1, min_values=[-5, -5], max_values=[5, 5], target_function=target_function):
    for i in range(position.shape[0]):
        # uniform random number between 0 and 1
        r1_leader = int.from_bytes(os.urandom(
            8), byteorder="big") / ((1 << 64) - 1)
        # uniform random number between 0 and 1
        r2_leader = int.from_bytes(os.urandom(
            8), byteorder="big") / ((1 << 64) - 1)
        a_leader = 2*a_linear_component*r1_leader - \
            a_linear_component  # uniform random number between -1 and 1
        c_leader = 2*r2_leader
        p = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
        for j in range(len(min_values)):
            if(p < 0.5):
                if(abs(a_leader) >= 1):
                    rand = int.from_bytes(os.urandom(
                        8), byteorder="big") / ((1 << 64) - 1)
                    rand_leader_index = math.floor(position.shape[0]*rand)
                    x_rand = position[rand_leader_index, :]
                    distance_x_rand = abs(c_leader*x_rand[j] - position[i, j])
                    position[i][j] = np.clip(
                        x_rand[j] - a_leader*distance_x_rand, min_values[j], max_values[j])
                elif(abs(a_leader) < 1):
                    # distance between leader and current position
                    distance_leader = abs(
                        c_leader*leader[0, j] - position[i, j])
                    position[i][j] = np.clip(
                        leader[0, j] - a_leader*distance_leader, min_values[j], max_values[j])
            elif(p >= 0.5):
                # distance between leader and current position
                distance_leader = abs(leader[0, j] - position[i, j])
                rand = int.from_bytes(os.urandom(
                    8), byteorder="big") / ((1 << 64) - 1)
                m_param = (b_linear_component - 1)*rand + 1
                position[i][j] = np.clip((distance_leader*math.exp(spiral_param*m_param)*math.cos(
                    m_param*2*math.pi)+leader[0, j]), min_values[j], max_values[j])
        position[i][-1] = target_function(position[i][:len(min_values)])
    return position


def whale_optimization_algorithm(hunting_party=10, spiral_param=1,  min_values=[-5, -5], max_values=[5, 5], iterations=50, step=50, target_function=target_function):
    count = 0
    hist = []
    leader_plot = []
    positions = whale_initial_position(
        hunting_party=hunting_party, min_values=min_values, max_values=max_values, target_function=target_function)
    leader = whale_leader_position(dimension=len(
        min_values), target_function=target_function)
    while(count < iterations):
        leader_plot.append(list(leader[0, :len(min_values)]))
        plt.show()
        hist.append(leader[0, -1])
        a_linear_component = 2 - count*(2/iterations)  # linear component of a
        b_linear_component = -1 + count * \
            (-1/iterations)  # linear component of b
        leader = whale_update_leader(positions, leader)
        positions = whale_update_position(positions, leader, a_linear_component=a_linear_component, b_linear_component=b_linear_component,
                                          spiral_param=spiral_param, min_values=min_values, max_values=max_values, target_function=target_function)
        count += 1
    return leader, hist, leader_plot


woa, woahist, woaleader_plot = whale_optimization_algorithm(
    target_function=beale)
print("WOA:", woa)

# plot the histogram of the WOA
plt.plot(woahist)
plt.show()
