import numpy as np
import pandas as pd
import math
import random
import os
import matplotlib.pyplot as plt
from test_functions import *


def euclidean_distence(x, y):
    return np.sqrt(np.sum((x-y)**2))


def hybrid_initital_position(hunting_party=5, min_values=[-5, -5], max_values=[5, 5], target_function=target_function):
    positions = np.zeros((hunting_party, len(min_values)+1))
    for i in range(hunting_party):
        for j in range(len(min_values)):
            positions[i][j] = random.uniform(min_values[j], max_values[j])
        positions[i][len(min_values)] = target_function(
            positions[i][:len(min_values)])
    return positions


def hybrid_whale_leader(dimension=2, target_function=target_function):
    leader = np.zeros((1, dimension+1))
    for i in range(dimension):
        leader[0][i] = random.uniform(-5, 5)
    leader[0][dimension] = target_function(leader[0][:dimension])
    return leader


def beta_value(x, y, gama=1, beta_0=1):
    rij = euclidean_distence(x, y)
    beta = beta_0*math.exp(-gama*(rij)**2)
    return beta


def hybrid_update_leader(positions, leader):
    for i in range(positions.shape[0]):
        if (leader[0, -1] > positions[i, -1]):
            for j in range(positions.shape[1]):
                leader[0, j] = positions[i, j]
    return leader


def hybrid_update_position(position, leader, spiral_param=1, a_linear_component=2, b_linear_component=1, alpha_0=0.2, beta_0=1, gama=1, min_values=[-5, -5], max_values=[5, 5], target_function=target_function):
    for i in range(position.shape[0]):
        r1_leader = int.from_bytes(os.urandom(
            8), byteorder="big") / ((1 << 64) - 1)
        # uniform random number between 0 and 1
        r2_leader = int.from_bytes(os.urandom(
            8), byteorder="big") / ((1 << 64) - 1)
        a_leader = 2*a_linear_component*r1_leader - a_linear_component
        c_leader = 2*r2_leader
        x = leader[0, :len(min_values)]
        y = position[i, :len(min_values)]
        p = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
        for j in range(len(min_values)):
            if(p < 0.5):
                if(abs(a_leader) < 1):
                    distance_leader = abs(
                        c_leader*leader[0, j] - position[i, j])
                    position[i][j] = np.clip(
                        leader[0, j] - a_leader*distance_leader, min_values[j], max_values[j])
                else:
                    rand = int.from_bytes(os.urandom(
                        8), byteorder="big") / ((1 << 64) - 1)
                    rand_leader_index = math.floor(position.shape[0]*rand)
                    x_rand = position[rand_leader_index, :]
                    distance_x_rand = abs(c_leader*x_rand[j] - position[i, j])
                    distance_leader = abs(
                        c_leader*leader[0, j] - position[i, j])
                    eta = 2 / (1+math.exp(beta_value(x, y, gama=gama, beta_0=beta_0)
                               * np.sign(np.sum(x-y))*np.linalg.norm(distance_leader)))
                    position[i][j] = np.clip(
                        x_rand[j] - a_leader*distance_x_rand*eta, min_values[j], max_values[j])
            else:
                if(abs(a_leader) < 1):
                    rand = int.from_bytes(os.urandom(
                        8), byteorder="big") / ((1 << 64) - 1)
                    m_param = (b_linear_component - 1)*rand + 1
                    distance_leader = abs(leader[0, j] - position[i, j])
                    eta1 = 2/(1+math.exp(b_linear_component*beta_value(x, y,
                              gama=gama, beta_0=beta_0)*np.linalg.norm(distance_leader)))
                    position[i][j] = np.clip(distance_leader*eta1*math.exp(spiral_param*m_param)*math.cos(
                        m_param*2*math.pi)+leader[0, j], min_values[j], max_values[j])
                else:
                    rand = int.from_bytes(os.urandom(
                        8), byteorder="big") / ((1 << 64) - 1)
                    rand_leader_index = math.floor(position.shape[0]*rand)
                    x_rand = position[rand_leader_index, :]
                    distance_x_rand = abs(c_leader*x_rand[j] - position[i, j])
                    distance_leader = abs(
                        c_leader*leader[0, j] - position[i, j])
                    eta2 = 2/(1+math.exp(b_linear_component*beta_value(x, y,
                              gama=gama, beta_0=beta_0)*np.linalg.norm(distance_x_rand)))
                    position[i][j] = np.clip(
                        x_rand[j] - a_leader*distance_x_rand*eta2, min_values[j], max_values[j])
        position[i][-1] = target_function(position[i][:len(min_values)])

    return position


def whalefireflyhybrid(huntingparty=5, spiral_param=1, min_values=[-5, -5], max_values=[5, 5], generations=50, alpha_0=0.2, beta_0=1, gama=1, target_function=target_function):
    count = 0
    hist = []
    positions = hybrid_initital_position(
        hunting_party=huntingparty, min_values=min_values, max_values=max_values, target_function=target_function)
    leader = hybrid_whale_leader(dimension=len(
        min_values), target_function=target_function)

    while(count <= generations):
        # print("Generation:",count,"Leader:",leader)
        hist.append(leader[0, -1])
        a_linear_component = 2 - count*(2/generations)  # linear component of a
        b_linear_component = -1 + count * \
            (-1/generations)  # linear component of b
        leader = hybrid_update_leader(positions, leader)
        positions = hybrid_update_position(positions, leader, spiral_param=spiral_param, a_linear_component=a_linear_component, b_linear_component=b_linear_component,
                                           alpha_0=alpha_0, beta_0=beta_0, gama=gama, min_values=min_values, max_values=max_values, target_function=target_function)
        count += 1
    return leader, hist
