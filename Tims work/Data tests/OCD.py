import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneOut
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


def OCD_path(id):
    return f'OCD-CON\\OCD\\a{id}_fc_mat.txt'

def CON_path(id):
    return f'OCD-CON\CON1\c{id}_fc_mat.txt'

OCD_ids = [113, 130, 146, 154, 159, 162, 164, 168, 215, 217, 222, 225, 226, 227, 228, 229, 230, 233, 235, 236, 237]

X = []
y = []
for id in OCD_ids:
    x = pd.read_csv(OCD_path(id), header = None, sep = '\n').iloc[:,:].values
    values = np.array([])
    for b in x:
        # print(np.append(values,b, axis = 0))
        values = np.append(values, b)

    X.append(values)
    y.append(1)

CON_ids = [105, 107, 112, 118, 119, 136, 137, 139, 148, 156, 161, 163, 171, 173, 174, 209, 211, 213, 219, 224]
for id in CON_ids:
    x = pd.read_csv(CON_path(id), header = None, sep = '\n').iloc[:,:].values
    values = np.array([])
    for b in x:
        # print(np.append(values,b, axis = 0))
        values = np.append(values, b)

    X.append(values)
    y.append(0)


# print(x)
# print(len(x), len(y))
# test_class = Distance_classifier(X,list(y), model = "gamma")
#
# test_class.fit()
#
# test_class.mle()
#
# print(test_class.score())
X = np.asarray(X)
y = np.asarray(y)

loo = LeaveOneOut()
acc = []
scores = []
no_class = 0
for train, test in loo.split(X):
    print(f"train is {train} and test is {test}")
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]

    test_class = Distance_classifier(X_train,list(y_train), model = "gamma")

    test_class.fit()

    # test_class.mle()
    # print(y_test[0], test_class.predict(X_test, explicit = False))
    predict = test_class.predict(X_test, explicit = False)
    # print(f"total score is : {test_class.score(explicit = True)}")
    scores.append(test_class.score(X_train, y_train, explicit = False))
    if y_test[0] == predict:
        acc.append(1)
    else:
        acc.append(0)
        if predict == -1:
            no_class += 1

print(f"average score is {np.mean(scores)} with {np.mean(acc)} accuracy")
