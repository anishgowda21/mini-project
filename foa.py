import numpy as np
import pandas as pd
import math
import random
import os
from scipy.special import gamma
import matplotlib.pyplot as plt
from test_functions import *
from numpy import array, sin, cos, exp, sqrt, pi, abs


def initial_fireflies(swarm_size=3, min_values=[-5, -5], max_values=[5, 5], target_function=target_function):
    position = np.zeros((swarm_size, len(min_values)+1))
    for i in range(swarm_size):
        for j in range(len(min_values)):
            position[i][j] = random.uniform(min_values[j], max_values[j])
        position[i][len(min_values)] = target_function(
            position[i][:len(min_values)])
    return position


def firefly_algorithm(swarm_size=5, min_values=[-5, -5], max_values=[5, 5], generations=50, alpha_0=0.2, beta_0=1, gama=1, target_function=target_function):
    count = 0
    hist = []
    positions = initial_fireflies(target_function=target_function)


firefly_algorithm(target_function=easom)
