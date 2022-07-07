from csv import writer
from test_functions import *
from woa import whale_optimization_algorithm
from foa import firefly_algorithm
import matplotlib.pyplot as plt
from whaleFFHybrid import whalefireflyhybrid
import os
import numpy as np


def whale_writer(data):
    with open("dataFiles/woa_func.csv", "a") as f:
        wo = writer(f)
        wo.writerow(data)
        f.close()


def ff_writer(data):
    with open("dataFiles/ff_func.csv", "a") as f:
        fo = writer(f)
        fo.writerow(data)
        f.close()


def hoa_writer(data):
    with open("dataFiles/hoa_func.csv", "a") as f:
        ho = writer(f)
        ho.writerow(data)
        f.close()


# Function automation to apply  optimization algorithms on all functions
def function_automation():
    fucs = [easom, beale, bukin_function, camel6, crossit, eggholder, holdertable,
            levy13, michalewicz, rosenbrock, schewefel, stybtang, mccormick_function]
    total = len(fucs)
    boundary_values_dictionary = {'easom': [[-5, -5], [5, 5]], 'beale': [[-5, -5], [5, 5]], 'bukin_function': [[-15, -15], [5, 5]], 'camel6': [[-5, -5], [5, 5]], 'crossit': [[-5, -5], [5, 5]], 'eggholder': [[-5, -5], [600, 600]], 'holdertable': [[-10, -10], [
        10, 10]], 'levy13': [[-5, -5], [5, 5]], 'michalewicz': [[-5, -5], [5, 5]], 'rosenbrock': [[-5, -5], [5, 5]], 'schewefel': [[-5, -5], [500, 500]], 'shubert': [[-6, -6], [6, 6]], 'stybtang': [[-5, -5], [5, 5]], 'mccormick_function': [[-5, -5], [5, 5]]}

    i = 0
    hp = 10
    para_m = 3
    iter = 30
    while(i < total):
        tf = mccormick_function
        minv = boundary_values_dictionary[tf.__name__][0]
        maxv = boundary_values_dictionary[tf.__name__][1]
        woa, woa_hist = whale_optimization_algorithm(
            hunting_party=hp, spiral_param=para_m, iterations=iter, target_function=tf, min_values=minv, max_values=maxv)
        ff, ff_hist = firefly_algorithm(
            swarm_size=5, generations=iter, target_function=tf, min_values=minv, max_values=maxv)
        hoa, hoa_hist = whalefireflyhybrid(
            huntingparty=5, spiral_param=1, generations=iter, target_function=tf, min_values=minv, max_values=maxv)
        plt.plot(range(iter), woa_hist[1:], label='woa')
        plt.plot(range(iter), ff_hist[1:], label='FOA')
        plt.plot(range(iter), hoa_hist[1:], label='HOA')
        plt.title(str(tf.__name__)+" function")
        plt.legend()
        plt.savefig("Images/"+str(tf.__name__)+" function.png")
        plt.show()

        woaList = woa[0].tolist()
        woaList.insert(0, tf.__name__)
        ffList = ff.tolist()
        ffList.insert(0, tf.__name__)
        hoaList = hoa[0].tolist()
        hoaList.insert(0, tf.__name__)
        whale_writer(woaList)
        ff_writer(ffList)
        hoa_writer(hoaList)
        i += 1


# Dataset automation to apply all optimization algorithms on all datasets
def dataset_automation():
    hp = 30
    para_m = 3
    iternum = 150
    dataset_cluster_values_dictionary = {'aggregation.csv': 7,
                                         'varied.csv': 3,
                                         'seeds.csv': 3,
                                         'jain.csv': 2,
                                         'heart.csv': 2,
                                         'blobs.csv': 3,
                                         'mouse.csv': 3,
                                         'appendicitis.csv': 2,
                                         'ionosphere.csv': 2,
                                         'Blood.csv': 2,
                                         'vary-density.csv': 3,
                                         'vertebral2.csv': 2,
                                         'balance.csv': 3,
                                         'wdbc.csv': 2,
                                         'liver.csv': 2,
                                         'banknote.csv': 2,
                                         'aniso.csv': 3,
                                         'flame.csv': 2,
                                         'iris.csv': 3,
                                         'smiley.csv': 4,
                                         'circles.csv': 2,
                                         'vertebral3.csv': 3,
                                         'iris2D.csv': 3,
                                         'diagnosis_II.csv': 2,
                                         'sonar.csv': 2,
                                         'moons.csv': 2,
                                         'pathbased.csv': 3,
                                         'wine.csv': 3,
                                         'glass.csv': 6,
                                         'ecoli.csv': 7}
    keyy = ['aggregation.csv', 'varied.csv', 'seeds.csv', 'jain.csv', 'heart.csv', 'blobs.csv', 'mouse.csv', 'appendicitis.csv', 'ionosphere.csv', 'Blood.csv', 'vary-density.csv', 'vertebral2.csv', 'balance.csv', 'wdbc.csv',
            'liver.csv', 'banknote.csv', 'aniso.csv', 'flame.csv', 'iris.csv', 'smiley.csv', 'circles.csv', 'vertebral3.csv', 'iris2D.csv', 'diagnosis_II.csv', 'sonar.csv', 'moons.csv', 'pathbased.csv', 'wine.csv', 'glass.csv', 'ecoli.csv']
    vals = [7, 3, 3, 2, 2, 3, 3, 2, 2, 2, 3, 2, 3, 2,
            2, 2, 3, 2, 3, 4, 2, 3, 3, 2, 2, 2, 3, 3, 6, 7]
    i = 0

    while(i < len(keyy)):
        file = keyy[i]
        k = vals[i]
        title = file.split('.')[0]
        min_, max_ = read_data(file, k)
        print_glob()
        woa, woa_hist = whale_optimization_algorithm(
            hunting_party=10, spiral_param=3, min_values=min_*k, max_values=max_*k, iterations=iternum, target_function=evaluate)
        print("\n")
        ff, ff_hist = firefly_algorithm(
            swarm_size=50, min_values=min_*k, max_values=max_*k, generations=iternum, target_function=evaluate)
        print("\n")
        hoa, hoa_hist = whalefireflyhybrid(huntingparty=10, spiral_param=3, min_values=min_*k,
                                           max_values=max_*k, generations=iternum, target_function=evaluate)
        print("\n")

        plt.plot(range(iternum), woa_hist[1:], label='woa')
        plt.plot(range(iternum), ff_hist[1:], label='FOA')
        plt.plot(range(iternum), hoa_hist[1:], label='HOA')
        plt.title(str(title)+" dataset")
        plt.legend()
        plt.savefig("Images/"+str(title)+" dataset.png")
        plt.show()

        woaList = woa[0].tolist()
        woaList.insert(0, title)
        ffList = ff.tolist()
        ffList.insert(0, title)
        hoaList = hoa[0].tolist()
        hoaList.insert(0, title)
        whale_writer(woaList)
        ff_writer(ffList)
        hoa_writer(hoaList)
        print("Saved "+str(title))
        i += 1
