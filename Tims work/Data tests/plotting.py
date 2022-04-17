import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import RadiusNeighborsClassifier
import matplotlib.pyplot as plt
import math
from collections import defaultdict
import operator
import scipy as sp
from sklearn import preprocessing
import json
from Distance_Classifier import Distance_classifier
from sklearn.metrics import r2_score

def show_plot(norm = True, power = 1/3, num = 8, probScale = "linear", cdfGraph = False, kd_tree = True, test = "biogrid", in_data = "",\
             X = None, y = None, decode = True):
    if decode:
        cats = num
        X = []
        y = []
        for i in range(cats):
            x = pd.read_csv(in_data[i], header = None, sep = ' ').iloc[:,:].values
            for b in x:
                X.append(b)
                y.append(i)
        y = np.asarray(y)
    # X = customScaling(np.asarray(X), scale = power)
    if norm == 'custom':
        pass
    elif norm == "Ignore":
        pass
    else:
        x = normalize(X) if norm else X
    test_class = Distance_classifier()
    test_class.fit(X,y)


    def scale(array):
        print(np.sum(array<0))
        print(np.sum(array[np.where(array != np.inf)]))
        return array/np.sum(array[np.where(array != np.inf)])

    for graph in ["gamma"]:
        #print("For", graph, "distribution the outcomes are:")
        total = 0
        for key in test_class.get_details():
            #print('for class', key)
        #     print(len(test_class.get_details()[key][key]))
            fig, ax1 = plt.subplots(1, 1)
            points = test_class.get_details()[key][key]
            n, bins, patches = ax1.hist(points, bins = 17)
            print(f'{bins[-2]}: {[b for b in patches][-1].get_height()}')
            ax1.set_ylabel('number', color="tab:red")
            ax1.tick_params(axis='y', labelcolor="tab:red")
            ax1.set_xlabel("distance", color = "black")
            ax1.tick_params(axis = 'x', labelcolor = "black")
            x = np.linspace(0,bins[-1],200)
            if graph == "gamma":
                alpha, theta = test_class.gamma_alphas[key]
    #             print(test_class.gamma_alphas[key])
        #         print(bins)
                rv = sp.stats.gamma(alpha ,scale = theta)
                pdf = rv.pdf(x)
                cdf = rv.cdf(x)
                cdf_in = 1 - rv.cdf(x)
            elif graph == "exp":
                alpha, theta = test_class.gamma_alphas[key]
                print("Alpha and theata are:", alpha, theta)
                print("mean with gamma is is:", theta/alpha)
                test_class.mle(model = "exp",iterations = 15)
                print("mean with exp is:", 1/test_class.lambdas[key])
                Lambda = 1/test_class.lambdas[key]
    #             print("the lambda is:", Lambda)
                pdf = Lambda * np.exp(-Lambda * x)
            '''
            Checking where outliers are

            for cla,data in test_class.debug.items():
                max_index = max(data.items(), key=operator.itemgetter(1))[0]
                while max_index - 499 > 0:
                    max_index -= 499
                print(f'maximum for class {cla}: {max_index}')
            '''

            ax2 = ax1.twinx()
    #         pdf = scale(pdf)
            if not cdfGraph:
                ax2.plot(x, pdf)
            else:
                ax2.plot(x, cdf)
                ax2.plot(x, cdf_in)
            ax2.set_yscale(probScale);
        #     print('max of pdf is:', np.max(pdf))
            print("saving")
            plt.savefig(f'{test} scale__{power} normalize__{norm} cat__{key} cdf__{cdfGraph} prob__{probScale} kd__{kd_tree}.png')

            #print("Lowest p-score is:", 1-rv.cdf(np.max(points)))
            low_pscore = np.sum(1-rv.cdf(points) < 1/np.size(points))
            #print("Points with low p-scores:", low_pscore)
            total += np.sum(1-rv.cdf(points) < 1/np.size(points))
            plt.show()
            plt.close()
        print("P scores less than 1/samples:", total)
        print()
        print()
        print()
        return total
    # print(test_class.gamma_alphas)
# show_plot()
