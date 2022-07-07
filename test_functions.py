from numpy import array, sin, cos, exp, sqrt, pi, abs
import math
from scipy.special import gamma
import numpy as np
import pandas as pd

# Test functions


def easom(variables_values=[0, 0]):
    return -math.cos(variables_values[0])*math.cos(variables_values[1])*math.exp(-(variables_values[0] - math.pi)**2 - (variables_values[1] - math.pi)**2)


def beale(variables_values=[0, 0]):
    tmp1 = np.power(
        1.5 - variables_values[0] + variables_values[0] * variables_values[1], 2)
    tmp2 = np.power(
        2.25 - variables_values[0] + variables_values[0] * np.power(variables_values[1], 2), 2)
    tmp3 = np.power(
        2.625 - variables_values[0] + variables_values[0] * np.power(variables_values[1], 3), 2)
    return tmp1+tmp2+tmp3


def levy13(variables_values=[0, 0]):
    temp1 = (sin(3 * pi * variables_values[0]) ** 2)
    temp2 = (variables_values[0] - 1) ** 2 * \
        (1 + (sin(3 * pi * variables_values[1])) ** 2)
    temp3 = (variables_values[1] - 1) ** 2 * \
        (1 + (sin(2 * pi * variables_values[1])) ** 2)

    return temp1 + temp2 + temp3


# McCormick function min =-1.9133 at [-0.547, -1.547]

def mccormick_function(variables_values=[0, 0]):
    tmp1 = np.sin(variables_values[0]+variables_values[1]) + \
        np.power(variables_values[0]-variables_values[1], 2)
    tmp2 = -1.5*variables_values[0]+2.5*variables_values[1]+1
    return tmp1+tmp2


def stybtang(variables_values=[0, 0]):
    d = len(variables_values)
    val = 0.0
    for i in range(d):
        xi = variables_values[i]
        new = xi ** 4 - 16 * xi ** 2 + 5 * xi
        val += new

    return val / 2


def holdertable(variables_values=[0, 0]):
    term1 = sin(variables_values[0]) * cos(variables_values[1])
    term2 = exp(
        abs(1 - sqrt(variables_values[0] ** 2 + variables_values[1] ** 2) / pi))

    return -abs(term1 * term2)


def camel6(variables_values=[0, 0]):
    term1 = (4 - 2.1 * variables_values[0] ** 2 +
             (variables_values[0] ** 4) / 3) * variables_values[0] ** 2
    term2 = variables_values[0] * variables_values[1]
    term3 = (-4 + 4 * variables_values[1] ** 2) * variables_values[1] ** 2

    return term1 + term2 + term3


def eggholder(variables_values=[0, 0]):
    temp1 = -(variables_values[1] + 47) * \
        sin(sqrt(abs(variables_values[1] + (variables_values[0] / 2) + 47)))
    temp2 = - variables_values[0] * \
        sin(sqrt(abs(variables_values[0] - (variables_values[1] + 47))))

    return temp1 + temp2


def shubert(variables_values=[0, 0]):
    sum1 = 0
    sum2 = 0
    for i in range(1, 6):
        new1 = i * cos((i + 1) * variables_values[0] + i)
        new2 = i * cos((i + 1) * variables_values[1] + i)
        sum1 += new1
        sum2 += new2

    return sum1 * sum2


def rosenbrock(variables_values=[0, 0]):
    d = len(variables_values)
    val = 0.0
    for i in range(d - 1):
        val += 100.0 * (variables_values[i + 1] - variables_values[i]
                        ** 2) ** 2 + (variables_values[i] - 1) ** 2

    return val


def michalewicz(variables_values=[0, 0]):
    val = 0.0
    m = 10
    for i in range(len(variables_values)):
        val += sin(variables_values[i]) * sin(((i + 1)
                                               * (variables_values[i] ** 2)) / pi) ** (2 * m)

    return -val


def bukin_function(variables_values=[0, 0]):
    tmp1 = 100 * \
        np.sqrt(np.absolute(
            variables_values[1]-0.01*np.power(variables_values[1], 2)))
    tmp2 = 0.01*np.absolute(variables_values[0]+10)
    return tmp1+tmp2


def crossit(variables_values=[0, 0]):
    term1 = sin(variables_values[0]) * sin(variables_values[1])
    term2 = exp(
        abs(100 - sqrt(variables_values[0] ** 2 + variables_values[1] ** 2) / pi))

    return -0.0001 * (abs(term1 * term2) + 1) ** 0.1


def schewefel(variables_values=[0, 0]):
    val = 0
    for x in variables_values:
        val = val + x * sin(sqrt(abs(x)))

    return 418.9829 * len(variables_values) - val


def target_function(params=[]):
    return


def read_data(file_name, k_val):
    global data, k
    data = pd.read_csv('./Datasets/'+file_name, header=None)
    y = data.iloc[:, -1].values
    data = data.iloc[:, :-1].values
    k = k_val
    print(len(data[0]))

    min_ = []
    max_ = []
    # print(data.shape)
    for i in range(data.shape[1]):
        d = [x[i] for x in data]
        min_.append(min(d))
        max_.append(max(d))

    return min_, max_


def calc_distance(X1, X2):
    return(sum((X1 - X2)**2))**0.5


def findClosestCentroids(ic):
    assigned_centroid = []
    for i in data:
        distance = []
        for j in ic:
            distance.append(calc_distance(i, j))
        assigned_centroid.append(np.argmin(distance))
    return assigned_centroid


def evaluate(sol):

    sol = np.array(sol).reshape(k, -1)
    assigned_centroids = findClosestCentroids(sol)
    # print("assigned centroids = ",assigned_centroids)
    output = 0
    for i in range(len(data)):
        output += calc_distance(data[i], sol[assigned_centroids[i]])
    return output


def print_glob():
    print("Global k = ", k)
