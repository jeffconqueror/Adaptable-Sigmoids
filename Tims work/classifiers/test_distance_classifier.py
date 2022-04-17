import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize
import math
from collections import defaultdict
import scipy as sp

class Distance_classifier(BaseEstimator,ClassifierMixin):
    #Need to add compatability where class labels are strings.

    def __init__(self, model = "gamma", threshold = .00, kd= False, outlier_constant=1.5):
        self.model = model
        self.threshold = threshold
        self.kd = kd
        self.outlier_constant=outlier_constant

    def _calc_distance(self,origin, other):
        return np.sum((origin - other) ** 2)**(1/2)
    
    def _distances(self, data):
        zeros = 0
        short_dist = defaultdict(int)
        for i, to_data in enumerate(self.data):
            expect_dist = self._calc_distance(data, to_data)
            if expect_dist != 0:
                    if short_dist[self.labels[i]] > expect_dist:
                        short_dist[self.labels[i]] = expect_dist
                    if short_dist[self.labels[i]] == 0:
                        short_dist[self.labels[i]] = expect_dist
            elif expect_dist == 0:
                zeros += 1
        return short_dist
    
    def _find_outliers(self,dataset):
        #defintion of outlier w/ 1.5 iqr definition
        upper_quartile = np.percentile(dataset, 75)
        lower_quartile = np.percentile(dataset, 25)
        IQR = (upper_quartile - lower_quartile) * self.outlier_constant
        outliers = dataset[dataset >= upper_quartile + IQR]
        non_outliers = dataset[dataset < upper_quartile + IQR]
        return outliers, non_outliers

    def _add_secondary(self):
        # creating secondary distribution for outliers
        self.secondary_dist = [None for i in range(len(self.labels))] #the secondary distribution is a expo dist
        for label in self.distance_.keys():
            #only need to find distance to same class
            distance_to_class = np.asarray(self.distance_[label][label])
            outliers, non_outliers = self._find_outliers(distance_to_class)
            if len(outliers) == 0: #if no outliers do not add a class
                pass
            else: #if there are outliers add a new class and remove the outliers
                self.secondary_dist[int(label)] = np.mean(outliers)
                # second [lowest_new_class] to make sure other code works
#                     lowest_new_class += 1 # be able to make a new class

    def fit(self, data = None, labels = None, test = True):
        if data is not None and labels is not None:
            if len(data)!=len(labels):
                raise ValueError("data and labels must be the same length")
            data=normalize(data)
            self.data = np.asarray(data)
            self.labels = np.asarray(labels)
            #order the data to fit with processing
            proper_order = np.unravel_index(np.argsort(self.labels, axis=None), self.labels.shape)
            self.data = self.data[proper_order]
            self.labels = self.labels[proper_order]
        elif data==None and labels==None:
            self.data=data
            self.labels=labels
        else:
            raise ValueError("data and labels must have values or both must be None")
        # store all the distances in format Actual Class: To Class: [closest distances]
        self.distance_ = defaultdict(lambda: defaultdict(list))
        ### KD tree impementation of distance
        if self.kd:
            self.trees_ = {}
            #create the KD tree for each class
            for label in set(self.labels):
                #print(self.labels == label)
                self.trees_[label] = KDTree(self.data[self.labels == label])

            '''
            Save the distance for each for debug purposes
            '''
            self.debug_ = defaultdict(dict)
            #from tree find the closest that is not the same point
            # should be able to optize later by sending groups of points
            for i in range(len(self.data)):
                for label, tree in self.trees_.items():
                    #set k to 2 so that if same point found, use the second point
                    dist, ind = tree.query([self.data[i]], k = 2)
                    if label == self.labels[i]: #if same class, there will be a point w/ dist 0, sef
                        self.distance_[self.labels[i]][label].append(dist[0][1])
                        self.debug_[label][i] = dist[0][1]
                    else:
                        self.distance_[self.labels[i]][label].append(dist[0][0])

        elif test:
            for i in range(len(self.data)):
                shortests = self._distances(self.data[i])
                for key, shortest in shortests.items():
                    self.distance_[self.labels[i]][key].append(shortest)

            # print(self.distance_)
            self._mle()
            self._add_secondary()
        return self
    
    def get_details(self):
        return self.distance_

    def get_gamma_alphas(self):
        return self.gamma_alphas

    def _mle(self, model = "", iterations = 5):

        def gamma_approx():
            #using Gamma(a,beta) not Gamma(alpha, theta)
            alphas = np.zeros((len(set(self.labels)), 2)) # 0 is k, 1 is theta
            x = np.zeros((len(set(self.labels)), 2)) #0 is np.log(np.mean(x)) 1 is np.mean(np.log(x))
            for cat in set(self.labels):
#                 print("Catigory:",self.distance_[cat][cat])
                #print(x, self.distance_)
                x[int(cat)][0] = np.log(np.mean(self.distance_[cat][cat]))
                x[int(cat)][1] = np.mean(np.log(self.distance_[cat][cat]))

            alphas[:,0] = .5/(x[:,0] - x[:,1])

            k = alphas[:,0]
            for i in range(iterations):
                digamma = sp.special.digamma(k)
                digamma_prime = sp.special.polygamma(1, k)
                k = 1/ (1/k + (x[:,1] - x[:,0] + np.log(k) - digamma)/(k**2*(1/k - digamma_prime)))
                #print("itermidiary step:", k)

            alphas[:, 0] = k
            alphas[:, 1] = np.exp(x[:, 0])/alphas[:, 0]
            return alphas

        if model == "gamma" or (model == "" and self.model == "gamma"): #[1]
            self.model = "gamma"

            self.gamma_alphas = gamma_approx()
            print("made the alphas")

        else:
            print("Model is not supported")

    def predict(self, data, model = ""):
        data=normalize(data)
        if model == "gamma" or (model == "" and self.model == "gamma"):
            y_pred=np.empty((len(data),), dtype=self.labels[0].dtype)
            for i in range(len(data)):
                min_dists = self._distances(data[i])
                theta = self.gamma_alphas[:,0]
                k = self.gamma_alphas[:,1]
                predictions = np.zeros((self.gamma_alphas.shape[0],1))

                # create the models
                models = [sp.stats.gamma(theta[int(a)], scale = k[int(a)]) for a in min_dists.keys()]

                for cat, dist in min_dists.items():
                    predictions[int(cat)] = 1 - models[int(cat)].cdf(dist)
                    # get prediction for each class

                """# Uncomment for secondary distribution
                    if self.secondary_dist[cat] != None:
                        m = 1/self.secondary_dist[cat]
                        secondary_pred = np.e**(-m * dist) # calc p value for expo funct
                        if secondary_pred > .5:
                            secondary_pred -= .5
                            # should not be too close for the secondary distribution

                        predictions[cat] = max([secondary_pred, predictions[cat]])"""

                y_pred[i] = np.argmax(predictions) 
                    #if predictions[np.argmax(predictions)]  >= 1/np.count(self.labels, np.argmax(predictions)) else -1
            return y_pred

    def predict_proba(self, data, model = ""):
        data=normalize(data)
        if model == "gamma" or (model == "" and self.model == "gamma"):
            probabilities=[]
            for i in range(len(data)):
                min_dists = self._distances(data)
                theta = self.gamma_alphas[:,0]
                k = self.gamma_alphas[:,1]
                predictions = np.zeros((self.gamma_alphas.shape[0],1))

                # create the models
                models = [sp.stats.gamma(theta[int(a)], scale = k[int(a)]) for a in min_dists.keys()]

                for cat, dist in min_dists.items():
                    predictions[int(cat)] = 1 - models[int(cat)].cdf(dist)
                    # get prediction for each class

                """# Uncomment for secondary distribution
                    if self.secondary_dist[cat] != None:
                        m = 1/self.secondary_dist[cat]
                        secondary_pred = np.e**(-m * dist) # calc p value for expo funct
                        if secondary_pred > .5:
                            secondary_pred -= .5
                            # should not be too close for the secondary distribution

                        predictions[cat] = max([secondary_pred, predictions[cat]])"""
                probabilities.append(predictions)
            return probabilities

    #REDUNDANT
    # def score(self, model = "", explicit = False):
    #     if explicit:
    #         all_data = []
    #     if model == "":
    #         if self.model == "gamma":
    #             total = 0
    #             correct = 0
    #             pred_6 = 0
    #             for i in range(len(self.data)):
    #                 predictions = self.predict(self.data[i])
    #                 if explicit:
    #                     all_data.append(predictions)
    #                 else:
    #                     predict = np.argmax(predictions) if predictions[np.argmax(predictions)] >= self.threshold else -1
    #                     # print(f"predicted {predict}, should have been {self.labels[i]} because {predictions}")
    #                     if predict == self.labels[i]:
    #                         correct += 1
    #                     else:
    #                         print(f"distances are {self._distances(self.data[i])}\npredicted class of {predict} when actual was {self.labels[i]}")
    #                         print(f"the predictions were {predictions}")
    #                     total += 1
    #             print(f"the gammas alphas were {self.gamma_alphas}")
    #             return all_data if explicit else correct/total




