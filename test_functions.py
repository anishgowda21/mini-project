from numpy import array, sin, cos, exp, sqrt, pi, abs
import math
from scipy.special import gamma
import numpy as np


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


def target_function(params=[]):
    return
