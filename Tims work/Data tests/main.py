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
from Distance_Classifier import Distance_classifier, distance_adjustment, set_power
from sklearn.metrics import r2_score
from plotting import show_plot
from scipy import special

def get_raw_p(array):
    return 1-(np.argsort(array)+1)/(len(array) + 1)

def get_emp_p(array, k, theta):
    dist = sp.stats.gamma(k, scale = theta)
    return 1-dist.cdf(array)

def get_frechet_p(array, a, s, m):
    return np.exp(-( ((array-m)/s)**(-a)))

def frechet_pdf(x, m, s, a):
    over_s = a/s
    middle = ((x-m)/s)**(-1-a)
    expo = -(((x-m)/s) ** (-a))
    return over_s * middle * np.exp(expo)

def gen_gamma_pdf(x, a, d, p):
    return (p/(a**d))* (x **(d-1)) * np.exp( -( (x/a)**p)) / special.gamma(d/p)

LOG = True
FULL = True
ORGANISM = "AT"
NUM_TO_NAME = {
    0: "ER",
    1: "ERDD",
    2: "GEO",
    3: "GEOGD",
    4: "HGG",
    5: "SF",
    6: "SFDD",
    7: "Sticky"
}


data_location_AT = [r"D:\Storage\Research\data\ATER",
                 r"D:\Storage\Research\data\ATERDD",
                 r"D:\Storage\Research\data\ATGEO",
                 r"D:\Storage\Research\data\ATGEOGD",
                 r"D:\Storage\Research\data\ATHGG",
                 r"D:\Storage\Research\data\ATSF",
                 r"D:\Storage\Research\data\ATSFDD",
                 r"D:\Storage\Research\data\ATSticky",
                 r"D:\Storage\Research\data\ATOriginal"]

data_location_CE = [r"D:\Storage\Research\data\CEER",
                 r"D:\Storage\Research\data\CEERDD",
                 r"D:\Storage\Research\data\CEGEO",
                 r"D:\Storage\Research\data\CEGEOGD",
                 r"D:\Storage\Research\data\CEHGG",
                 r"D:\Storage\Research\data\CESF",
                 r"D:\Storage\Research\data\CESFDD",
                 r"D:\Storage\Research\data\CESticky",
                 r"D:\Storage\Research\data\CEOriginal"]

data_location_DM = [r"D:\Storage\Research\data\DMER",
                 r"D:\Storage\Research\data\DMERDD",
                 r"D:\Storage\Research\data\DMGEO",
                 r"D:\Storage\Research\data\DMGEOGD",
                 r"D:\Storage\Research\data\DMHGG",
                 r"D:\Storage\Research\data\DMSF",
                 r"D:\Storage\Research\data\DMSFDD",
                 r"D:\Storage\Research\data\DMSticky",
                 r"D:\Storage\Research\data\DMOriginal"]

data_location_EC = [r"D:\Storage\Research\data\ECER",
                 r"D:\Storage\Research\data\ECERDD",
                 r"D:\Storage\Research\data\ECGEO",
                 r"D:\Storage\Research\data\ECGEOGD",
                 r"D:\Storage\Research\data\ECHGG",
                 r"D:\Storage\Research\data\ECSF",
                 r"D:\Storage\Research\data\ECSFDD",
                 r"D:\Storage\Research\data\ECSticky",
                 r"D:\Storage\Research\data\ECOriginal"]

data_location_HS = [r"D:\Storage\Research\data\HSER",
                 r"D:\Storage\Research\data\HSERDD",
                 r"D:\Storage\Research\data\HSGEO",
                 r"D:\Storage\Research\data\HSGEOGD",
                 r"D:\Storage\Research\data\HSHGG",
                 r"D:\Storage\Research\data\HSSF",
                 r"D:\Storage\Research\data\HSSFDD",
                 r"D:\Storage\Research\data\HSSticky",
                 r"D:\Storage\Research\data\HSOriginal"]

data_location_RN = [r"D:\Storage\Research\data\RNER",
                 r"D:\Storage\Research\data\RNERDD",
                 r"D:\Storage\Research\data\RNGEO",
                 r"D:\Storage\Research\data\RNGEOGD",
                 r"D:\Storage\Research\data\RNHGG",
                 r"D:\Storage\Research\data\RNSF",
                 r"D:\Storage\Research\data\RNSFDD",
                 r"D:\Storage\Research\data\RNSticky",
                 r"D:\Storage\Research\data\RNOriginal"]

data_location_SC = [r"D:\Storage\Research\data\SCER",
                 r"D:\Storage\Research\data\SCERDD",
                 r"D:\Storage\Research\data\SCGEO",
                 r"D:\Storage\Research\data\SCGEOGD",
                 r"D:\Storage\Research\data\SCHGG",
                 r"D:\Storage\Research\data\SCSF",
                 r"D:\Storage\Research\data\SCSFDD",
                 r"D:\Storage\Research\data\SCSticky",
                 r"D:\Storage\Research\data\SCOriginal"]

data_location_SP = [r"D:\Storage\Research\data\SPER",
                 r"D:\Storage\Research\data\SPERDD",
                 r"D:\Storage\Research\data\SPGEO",
                 r"D:\Storage\Research\data\SPGEOGD",
                 r"D:\Storage\Research\data\SPHGG",
                 r"D:\Storage\Research\data\SPSF",
                 r"D:\Storage\Research\data\SPSFDD",
                 r"D:\Storage\Research\data\SPSticky",
                 r"D:\Storage\Research\data\SPOriginal"]

data_locations ={
                 "AT":data_location_AT#,
                 # "CE": data_location_CE,
                 # "DM": data_location_DM,
                 # "EC": data_location_EC,
                 # "HS": data_location_HS,
                 # "RN": data_location_RN,
                 # "SC": data_location_SC,
                 # "SP": data_location_SP
}
for ORGANISM, location in data_locations.items():
    num = 8
    cats = num if num <= 8 else 8

    X = []
    y = []
    for i in range(cats):
            x = pd.read_csv(location[i], header = None, sep = ' ').iloc[:,:].values
            for b in x:
                X.append(b)
                y.append(i)
    x = X

    X = normalize(X)
    print(np.median(x))
    x_train, x_test, y_train, y_test = train_test_split(X,y)

    for power in [0]:
        set_power(power)
            # for full test just use X and y
        if FULL:
            test_class = Distance_classifier(model = "gamma", threshold = 1/len(x_train))
            test_class.fit(X,list(y))
        else:
            test_class = Distance_classifier(x_train, list(y_train))
            test_class.fit(x_train, list(y_train))

        # show_plot(norm = "Ignore", power = 1, cdfGraph = False, test = "Biogrid log", X = X, y = y, decode = False)

        # print(gamma_alphas)

        print("values")

        '''
        Change to logs b/c logs are wonky
        '''
        for key in test_class.get_details():
            test_class.distance[key][key] = np.log(test_class.distance[key][key])
            test_class.distance[key][key] -= (np.min(test_class.distance[key][key]) - .00001)
        '''
        then refit
        '''
        test_class.__mle__()
        print("refitted")
        for key in test_class.get_details():
            dist = test_class.get_details()[key]
            #print('for class', key)
        #     print(len(test_class.get_details()[key][key]))
            fig, ax1 = plt.subplots(1, 1)

            points = test_class.get_details()[key][key]
            sorted_points = np.sort(points)
            k, θ = test_class.get_params()[key]
            gamma_p = get_emp_p(sorted_points, k, θ)
            actual_p = get_raw_p(sorted_points)

            print(np.max(actual_p), np.min(actual_p))
            # print((np.max(actual_p), np.max(gamma_p)))
            # x = np.linspace(0,(np.max(np.max(actual_p), np.max(gamma_p))),200)
            # print(np.argsort(gamma_p))
            # print(np.argsort(actual_p))
            ax1.plot(actual_p,actual_p)
            ax1.set_yscale("log")
            ax1.set_xscale('log')
            # ax2.plot(x, pdf, c = "red")
            ax1.plot( actual_p,gamma_p, c = "black", alpha = .75)
            plt.xlabel("actual p value")
            plt.ylabel("gamma predicted p value")
            # ax2.set_ysc0ale('log')
            # ax2.set_xscale('log')
            # plt.show()
            # plt.savefig(f"iris_class_{key}_p_value compare-{power}.png")
            # # plt.clf()
            # plt.close()

            # points = test_class.get_details()[key][key]
            # n, bins, patches = ax1.hist(points, bins = 17)
            # # print(f'{bins[-2]}: {[b for b in patches][-1].get_height()}')
            # ax1.set_ylabel('number', color="tab:red")
            # ax1.tick_params(axis='y', labelcolor="tab:red")
            # ax1.set_xlabel("distance", color = "black")
            # ax1.tick_params(axis = 'x', labelcolor = "black")
            #
            # # a,d,p = test_class.gen_gamma_params[key]
            # k, θ = test_class.gamma_alphas[key]
            # x = np.linspace(0,bins[-1],200)
            # # print(a,d,p)
            # # pdf = gen_gamma_pdf(x, a, d, p)
            # gamma_pdf = get_emp_p(x, k, θ)
            #
            # # ax2.plot(x, pdf, c = "red")
            # ax3.plot(x, gamma_pdf, c = "black", alpha = .5)
            # # ax2.set_yscale('log')
            # plt.show()
            plt.savefig(f"{ORGANISM}_{NUM_TO_NAME[key]}_p value compare- (1-log).png")
            # plt.clf()
            plt.close()


# details = test_class.get_details()
#
# for i in details.keys():
#     details[i] = np.asarray(details[i][i])
# #
# actual_p = {}
# distri_p = {}
# frechet_params = test_class.get_params()
#
# #
# for cat, dist in details.items():
#
#     actual_p[cat] = 1-np.asarray(get_raw_p(np.sort(dist)))
#     distri_p[cat] = 1-np.asarray(get_frechet_p(np.sort(dist), frechet_params[cat,0], frechet_params[cat,1], frechet_params[cat,2]))
#
# for cat in actual_p.keys():
# #     # for cdf in ["actual", "theory"]:
# #     #     np.savetxt(f"{ORGANISM}_{NUM_TO_NAME[cat]}_{cdf}.txt", actual_p[cat] if cdf == "actual" else distri_p[cat])
# #
# #         #plot the distributions
#     plt.plot(actual_p[cat], distri_p[cat])
# #
# #
# #     #put r^2 of line
#     plt.text(0.1,.5, f"Has a r^2 of {r2_score(actual_p[cat],distri_p[cat])}")
#     plt.text(0.01,.5, f"{ORGANISM} {NUM_TO_NAME[cat]} No Log")
# #
# #     if LOG:
#     plt.yscale('log')
#     plt.xscale('log')
# #
# #     #Label axis
#     plt.xlabel("Emprical CDF")
#     plt.ylabel("Theoretical CDF")
# #
# #     #plot y = x
#     plt.plot([0,1], [0,1])
# #
# #     plt.savefig(f"{ORGANISM}_log_{NUM_TO_NAME[cat]}_10-22 no log.png")
#     plt.show()
#     plt.clf()
# #
# #
