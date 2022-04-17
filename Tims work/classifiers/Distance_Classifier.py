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
from optimization import frechet_BFGS, gen_gamma_BFGS


power = 1
def distance(origin, other):
    return np.sum((origin - other) ** 2)**(1/2)

def read_data(path, header = None, seperator = ' '):
    return pd.read_csv(path, header = header, sep = seperator).iloc[:,:].values

def volume(dimensions, radius):
    return np.pi**(dimensions/2) * radius ** dimensions/ sp.special.gamma(dimensions/2 +1)

def customScaling(distances, scale = (1/3)):
    return distances ** scale

def distance_adjustment(distance):
    global power
    return distance

def set_power(to_set):
    global power
    power = to_set

class Distance_classifier():

    def __init__(self, model = "gamma", threshold = .05, kd_tree = False):
        self.kd = kd_tree
        # print([i for i in range(35) if i in y])
        self.model = model
        self.threshold = threshold
        self.count = 0

    def set_threshold(threshold):
        self.threshold = threshold

    def __distances__(self, data):
        zeros = 0
        short_dist = {}
        for i, to_data in enumerate(self.data):
            expect_dist = distance(data, to_data)
            if expect_dist != 0:
                if self.labels[i] in short_dist:
                    if short_dist[self.labels[i]] > distance_adjustment(expect_dist):
                        short_dist[self.labels[i]] = distance_adjustment(expect_dist)
                else:
                    short_dist[self.labels[i]] = distance_adjustment(expect_dist)
            elif expect_dist == 0:
                zeros += 1
        # if zeros != 1:
        #     print("found", zeros, "points with distance of 0")
        return short_dist

    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError
        else:
            self.data = np.asarray(X)
            self.labels = np.asarray(y)

            proper_order = np.unravel_index(np.argsort(self.labels, axis=None), self.labels.shape)
            self.data = self.data[proper_order]
            self.labels = self.labels[proper_order]

        def find_outliers(dataset, outlier_constant = 1.5):
            #defintion of outlier w/ 1.5 iqr definition
            upper_quartile = np.percentile(dataset, 75)
            lower_quartile = np.percentile(dataset, 25)
            IQR = (upper_quartile - lower_quartile) * outlier_constant
            outliers = dataset[dataset >= upper_quartile + IQR]
            non_outliers = dataset[dataset < upper_quartile + IQR]
            #print(outliers)
            return outliers, non_outliers

        def add_secondary():
            # creating secondary distribution for outliers
            self.secondary_dist = [None for i in range(len(self.labels))] #the secondary distribution is a expo dist
            for label in self.distance.keys():
                #only need to find distance to same class
                distances = np.asarray(self.distance[label][label])
                outliers, non_outliers = find_outliers(distances)
                if len(outliers) == 0: #if no outliers do not add a class
                    pass
                else: #if there are outliers add a new class and remove the outliers
                    self.secondary_dist[label] = np.mean(outliers)
                    # second [lowest_new_class] to make sure other code works
#                     lowest_new_class += 1 # be able to make a new class


        # store all the distances in format Actual Class: To Class: [closest distances]
        self.distance = defaultdict(lambda: defaultdict(list))


        ### KD tree impementation of distance
        ### not yet functional
        if self.kd:
            self.trees = {}
            #create the KD tree for each class
            for label in set(self.labels):
                ##print(self.labels == label)
                self.trees[label] = KDTree(self.data[self.labels == label])

            '''
            Save the distance for each for debug purposes
            '''
            self.debug = defaultdict(dict)
            #from tree find the closest that is not the same point
            # should be able to optize later by sending groups of points
            for i in range(len(self.data)):
                for label, tree in self.trees.items():
                    #set k to 2 so that if same point found, use the second point
                    dist, ind = tree.query([self.data[i]], k = 2)
                    if label == self.labels[i]: #if same class, there will be a point w/ dist 0, sef
                        self.distance[self.labels[i]][label].append(dist[0][1])
                        self.debug[label][i] = dist[0][1]
                    else:
                        self.distance[self.labels[i]][label].append(dist[0][0])

        for i in range(len(self.data)):
            shortests = self.__distances__(self.data[i])
            for key, shortest in shortests.items():
                self.distance[self.labels[i]][key].append(shortest)

            # #print(self.distance)
        self.__mle__()
        # add_secondary()

    def get_details(self):
        return self.distance

    def get_params(self):
        # returns the gamma alphas
        if self.model == "gamma":
            return self.gamma_alphas
        elif self.model == "frechet":
            return self.frechet_params

    def __mle__(self, model = "", iterations = 5):

        def gamma_approx():
            #using Gamma(shape,scale) not Gamma(shape, rate)
            alphas = np.zeros((len(set(self.labels)), 2)) # 0 is k, 1 is theta
            x = np.zeros((len(set(self.labels)), 2)) #0 is np.log(np.mean(x)) 1 is np.mean(np.log(x))
            for cat in set(self.labels):
#                 #print("Catigory:",self.distance[cat][cat])
                # print(x, self.distance)
                x[cat][0] = np.log(np.mean(self.distance[cat][cat]))
                x[cat][1] = np.mean(np.log(self.distance[cat][cat]))


            alphas[:,0] = .5/(x[:,0] - x[:,1])


            k = alphas[:,0]
            for i in range(iterations):
                digamma = sp.special.digamma(k)
                digamma_prime = sp.special.polygamma(1, k)
                k = 1/ (1/k + (x[:,1] - x[:,0] + np.log(k) - digamma)/(k**2*(1/k - digamma_prime)))
                ##print("itermidiary step:", k)

            alphas[:, 0] = k
            alphas[:, 1] = np.exp(x[:, 0])/alphas[:, 0]
            return alphas

        def frechet_approx():
            params = np.zeros((len(set(self.labels)), 3))
            for cat in set(self.labels):
                # print(self.distance[cat][cat])
                optimizer = frechet_BFGS(self.distance[cat][cat])
                params[cat] = optimizer.aprox(np.asarray([1e-5,.3,-1]))

            return params

        def gen_gamma_approx():
            approx_params = gamma_approx() #gives us [shape, scale]
            self.gamma_alphas = approx_params
            params = np.zeros((len(set(self.labels)), 3))
            #get the gamma aproximation for a starting point to find generalized gamma
            for cat in set(self.labels):
                optimizer = gen_gamma_BFGS(self.distance[cat][cat])
                params[cat] = optimizer.aprox(np.asarray([approx_params[cat][1],approx_params[cat][0],1]))

            print(f"gamma approx params: {approx_params}\ngen gamma params: {params}")
            return params

        if model == "gamma" or (model == "" and self.model == "gamma"): #[1]
            self.model = "gamma"

            self.gamma_alphas = gamma_approx()
            #print("made the alphas")
        elif model == "frechet" or (model == "" and self.model == "frechet"):
            self.model = "frechet"

            self.frechet_params = frechet_approx()

        elif model == "gen_gamma" or (model == "" and self.model == "gen_gamma"):
            self.model = "gen_gamma"
            self.gen_gamma_params = gen_gamma_approx()

            print(f"the calculated values are {self.gen_gamma_params}")

        else:
            print("Model is not supported")

    def predict(self, data, model = "", explicit = True): #change explicit to different variable name
        if model == "gamma" or (model == "" and self.model == "gamma"):

            min_dists = self.__distances__(data)
            if self.count < 5:
                # print(f"min dists is {min_dists}")
                self.count += 1
            theta = self.gamma_alphas[:,0]
            k = self.gamma_alphas[:,1]
            predictions = np.zeros((self.gamma_alphas.shape[0],1))
            # print(f"The gamma alpahs are {self.gamma_alphas} and the min distances are {min_dists}")

            # create the models
            models = [sp.stats.gamma(theta[a], scale = k[a]) for a in min_dists.keys()]

            for cat, dist in min_dists.items():
                predictions[cat] = 1 - models[cat].cdf(dist)
                # get prediction for each class

            """# Uncomment for secondary distribution
                if self.secondary_dist[cat] != None:
                    m = 1/self.secondary_dist[cat]
                    secondary_pred = np.e**(-m * dist) # calc p value for expo funct
                    if secondary_pred > .5:
                        secondary_pred -= .5
                        # should not be too close for the secondary distribution

                    predictions[cat] = max([secondary_pred, predictions[cat]])"""

            if not explicit:
                prediction = np.argmax(predictions) if predictions[np.argmax(predictions)] > self.threshold else -1
            return predictions if explicit else prediction


        elif model == "frechet" or (model == "" and self.model == "frechet"):
            min_dists = self.__distances__(data)
            m = self.frechet_params[:,0]
            s = self.frechet_params[:,1]
            predictions = np.zeros((self.frechet_params.shape[0],1))

            for cat, dist in min_dists.items():
                predictions[cat] = 1 - np.exp((-(dist - m[cat])/s[cat])**(-1))
                #we are using using frechet with alpha = 1
            if not explicit:
                prediction = np.argmax(predictions) if predictions[np.argmax(predictions)] > self.threshold else -1
            return predictions if explicit else prediction

    def score(self, X = None, y = None, model = "", explicit = False):
        if explicit:
            all_data = []
        if model == "":
            if self.model == "gamma":
                total = 0
                correct = 0
                # pred_6 = 0
                if not isinstance(X, np.ndarray) and not isinstance(y, np.ndarray):
                    for i in range(len(self.data)):
                        predictions = self.predict(self.data[i])
                        if explicit:
                            all_data.append(predictions)
                        else:
                            predict = np.argmax(predictions) if predictions[np.argmax(predictions)] > self.threshold else -1
                            # #print(f"predicted {predict}, should have been {self.labels[i]} because {predictions}")
                            if predict == self.labels[i]:
                                correct += 1
                            else:
                                pass
                                # #print(f"distances are {self.__distances__(self.data[i])}\npredicted class of {predict} when actual was {self.labels[i]}")
                                # #print(f"the predictions were {predictions}")
                            total += 1
                    # #print(f"the gammas alphas were {self.gamma_alphas}")
                    return all_data if explicit else correct/total

                else:
                    if len(X) != len(y):
                        raise NameError
                    for i in range(len(X)):
                        predictions = self.predict(X[i])
                        if explicit:
                            all_data.append(predictions)
                        else:
                            predict = np.argmax(predictions) if predictions[np.argmax(predictions)] > self.threshold else -1
                            if predict == y[i]:
                                correct += 1
                            total += 1
                    return all_data if explicit else correct/total


'''
Disregard this part, it just trys to streamline process of
making multiple distance classifiers and staking them
into a single classifier
'''
class distance_stack:
    def __init__(self, models, data_adjustment_funct):
        self.models = models
        self.functs = data_adjustment_funct
        if len(self.models) != len(self.functs):
            raise NameError

    def fit(self, X = None, y = None):
        # print(X)
        for index, model in enumerate(self.models):
            # print(len(X), len(y))
            model.fit(self.functs[index](X), y)
            # print(self.functs[index](X))
            model.mle()
            print(model.gamma_alphas)

    def predict(self, data):
        vote_min_p = defaultdict(float)
        votes = defaultdict(int)
        # probabilities
        for index, model in enumerate(self.models):
            vote = model.predict(self.functs[index](data))
            for index, prob in enumerate(vote):
                if vote_min_p[index] < prob:
                    vote_min_p[index] = prob
                    print(f'changed {index}')
            votes[model.predict(self.functs[index](data), explicit = False)] += 1
        print(vote_min_p, votes)
        return max(vote_min_p, key=vote_min_p.get)
