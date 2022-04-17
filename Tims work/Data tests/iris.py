import sys
sys.path.append('../')

import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
from collections import defaultdict
import operator
import scipy as sp
from sklearn import preprocessing
import json
from Distance_Classifier import Distance_classifier, distance_adjustment, set_power
from sklearn.metrics import r2_score
from plotting import show_plot
from scipy import special

def get_raw_p(array):
    return 1 - (np.argsort(array)+1)/(len(array) + 1)

def get_emp_p(array, k, theta):
    dist = sp.stats.gamma(k, scale = theta)
    return 1 - dist.cdf(array)

def get_frechet_p(array, a, s, m):
    return np.exp(-( ((array-m)/s)**(-a)))

def frechet_pdf(x, m, s, a):
    over_s = a/s
    middle = ((x-m)/s)**(-1-a)
    expo = -(((x-m)/s) ** (-a))
    return over_s * middle * np.exp(expo)

def gen_gamma_pdf(x, a, d, p):
    return (p/(a**d))* (x **(d-1)) * np.exp( -( (x/a)**p)) / special.gamma(d/p)

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

print(y)

test_class = Distance_classifier(model = "gamma")
for power in [1/4, 1/3, 1/2, 2/3, 3/4,1,2, 4/3, 3/2, 3,4]:
    set_power(power)
    test_class.fit(X,y)
    for key in test_class.get_details():
        dist = test_class.get_details()[key]
        #print('for class', key)
    #     print(len(test_class.get_details()[key][key]))
        fig, ax1 = plt.subplots(1, 1)
        # ax2 = ax1.twinx()
        # ax3 = ax2.twinx()

        points = test_class.get_details()[key][key]
        # n, bins, patches = ax1.hist(points, bins = 17)
        # print(f'{bins[-2]}: {[b for b in patches][-1].get_height()}')
        # ax1.set_ylabel('number', color="tab:red")
        # ax1.tick_params(axis='y', labelcolor="tab:red")
        # ax1.set_xlabel("distance", color = "black")
        # ax1.tick_params(axis = 'x', labelcolor = "black")

        # a,d,p = test_class.gen_gamma_params[key]
        k, θ = test_class.gamma_alphas[key]
        # print(a,d,p)
        # pdf = gen_gamma_pdf(x, a, d, p)
        points = test_class.get_details()[key][key]
        sorted_points = np.sort(points)
        gamma_p = get_emp_p(sorted_points, k, θ)
        actual_p = get_raw_p(sorted_points)

        print(np.max(actual_p), np.min(actual_p))
        # print((np.max(actual_p), np.max(gamma_p)))
        # x = np.linspace(0,(np.max(np.max(actual_p), np.max(gamma_p))),200)
        # print(np.argsort(gamma_p))
        # print(np.argsort(actual_p))
        ax1.plot(actual_p,actual_p)
        # ax2.plot(x, pdf, c = "red")
        ax1.plot(actual_p, gamma_p, c = "black", alpha = .5)
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        plt.xlabel("actual p score")
        plt.ylabel("gamma predicteed p score")
        # plt.show()
        plt.savefig(f"iris_class_{key}_p_value compare-{power}__1.png")
        # plt.clf()
        plt.close()
