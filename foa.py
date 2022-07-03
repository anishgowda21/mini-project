import numpy as np
import pandas as pd
import math
import random
import os
import matplotlib.pyplot as plt
from test_functions import *


def initial_fireflies(swarm_size=3, min_values=[-5, -5], max_values=[5, 5], target_function=target_function):
    position = np.zeros((swarm_size, len(min_values)+1))
    for i in range(swarm_size):
        for j in range(len(min_values)):
            position[i][j] = random.uniform(min_values[j], max_values[j])
        position[i][len(min_values)] = target_function(
            position[i][:len(min_values)])
    return position


def euclidean_distence(x, y):
    return np.sqrt(np.sum((x-y)**2))


def light_value(light_0, x, y, gama=1):
    rij = euclidean_distence(x, y)
    light = light_0*math.exp(-gama*(rij)**2)
    return light


def beta_value(x, y, gama=1, beta_0=1):
    rij = euclidean_distence(x, y)
    beta = beta_0*math.exp(-gama*(rij)**2)
    return beta


def firefly_update_position(position, x, y, alpha_0=0.2, beta_0=1, gama=1, firefly=0, min_values=[-5, -5], max_values=[5, 5], target_function=target_function):
    for j in range(len(x)):
        epson = int.from_bytes(os.urandom(
            8), byteorder="big") / ((1 << 64) - 1)
        position[firefly, j] = np.clip((x[j]+beta_value(x, y, gama=gama, beta_0=beta_0)*(
            y[j]-x[j])+alpha_0*epson), min_values[j], max_values[j])
    position[firefly, -
             1] = target_function(position[firefly, 0:position.shape[1]-1])
    return position


def firefly_algorithm(swarm_size=3, min_values=[-5, -5], max_values=[5, 5], generations=50, alpha_0=0.2, beta_0=1, gama=1, target_function=target_function):
    count = 0
    hist = []
    positions = initial_fireflies(swarm_size=swarm_size, min_values=min_values,
                                  max_values=max_values, target_function=target_function)
    while(count <= generations):
        # print("Generation :",count,"f(x) = ",positions[positions[:,-1].argsort()][0,:][-1])
        for i in range(swarm_size):
            for j in range(swarm_size):
                if(i != j):
                    firefly_i = np.copy(positions[i, 0:len(min_values)])
                    firefly_j = np.copy(positions[j, 0:len(min_values)])
                    light_i = light_value(
                        positions[i, -1], firefly_i, firefly_j, gama)
                    light_j = light_value(
                        positions[j, -1], firefly_j, firefly_i, gama)
                    if(light_i > light_j):
                        positions = firefly_update_position(positions, firefly_i, firefly_j, alpha_0=alpha_0, beta_0=beta_0,
                                                            gama=gama, firefly=i, min_values=min_values, max_values=max_values, target_function=target_function)

        count += 1
        best_firefly = np.copy(positions[positions[:, -1].argsort()][0, :])
        hist.append(best_firefly[-1])
    return best_firefly, hist
