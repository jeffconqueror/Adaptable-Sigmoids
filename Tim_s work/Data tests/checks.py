import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier
import matplotlib.pyplot as plt
import math
from collections import defaultdict
import operator
import scipy as sp
from sklearn import preprocessing
import json
from Distance_Classifier import Distance_classifier
from sklearn.metrics import r2_score
from sklearn.utils import shuffle


def get_raw_p(array):
    return np.argsort(array)/len(array)

def get_emp_p(array, k, theta):
    dist = sp.stats.gamma(theta, scale = k)
    return dist.cdf(array)


def get_data(path = "leaf.csv", y_label = "Class (Species)", remove = None):
    leaf_path = path
    df = pd.read_csv(leaf_path)
    if remove:
        df = df.loc[:, df.columns != remove]

    df_y = df[[y_label]]
    df_X = df.loc[:, df.columns != y_label]

    y = df_y.to_numpy().reshape(1, len(df_y))[0]
    X = df_X.to_numpy()
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    X, y = shuffle(X, y, random_state = 40061476)
    X = normalize(X)

    return X, y

def k_folds_test(X, y, compare_model = KNeighborsClassifier):


    kf = KFold(n_splits = 10, shuffle = True, random_state = 2020)
    scores_dc = []
    scores_knn = []

    kn = compare_model()
    dist_class = Distance_classifier()

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # print(train_index, test_index)
        dist_class.fit(X_train, y_train)
        kn.fit(X_train, y_train)
        # print(dist_class.predict(X[15]), dist_class.predict(X[15], explicit = False))
        # print(kn.predict([X[15]]), y[15])
        scores_dc.append(dist_class.score(X_test, y_test))
        scores_knn.append(kn.score(X_test, y_test))


    print(f"Scores dc: {scores_dc}, Avg Score: {np.mean(scores_dc)}")
    print(f"Scores {compare_model}: {scores_knn}, Avg Score: {np.mean(scores_knn)}")

def pdf_compare(data, X, y, log = False):
    dist_class = Distance_classifier()
    dist_class.fit(X, y)

    gamma_alphas = dist_class.get_params()
    details = dist_class.get_details()
    for i in details.keys():
        details[i] = np.asarray(details[i][i])

    actual_p = {}
    distri_p = {}

    for cat, dist in details.items():
        if len(dist) >=30:
            actual_p[cat] = 1 - np.asarray(get_raw_p(np.sort(dist)))
            distri_p[cat] = 1 - np.asarray(get_emp_p(np.sort(dist), gamma_alphas[cat,1], gamma_alphas[cat,0]))

            plt.plot(actual_p[cat], distri_p[cat])
            plt.xlabel("emprical")
            plt.ylabel("Theory")

            if log:
                plt.yscale('log')
                plt.xscale('log')

            plt.text(0,1, f"Has a r^2 of {r2_score(actual_p[cat],distri_p[cat])}")
            plt.plot([0,1], [0,1])

            plt.savefig(f"pdf comparision for {data} class {cat} log {log}.png")
            plt.clf()
