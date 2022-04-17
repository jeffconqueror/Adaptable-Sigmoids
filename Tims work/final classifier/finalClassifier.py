### Classifier ###
from sklearn.preprocessing import LabelEncoder
from helpers import gamma_mle
from collections import defaultdict
from nearest_neighbors import LinearNNSearch # TODO: integrate nearest neighbor search code w/ classifier
import scipy as sp
import numpy as np

class NNClassifier():
    def __init__(self, ε : float = 0.0001, threshold : float= .01):
        """
        :param ε: float
        :param threshold: float
        """
        self.ε = ε
        self.threshold = threshold

    def fit(self,X,y):
        # save the inputs into the model
        # also make sure it is a numpy array to make things easier
        self.input_data = np.asarray(X)

        # use label encoder to transform labels into integers
        # from 0->n where n is the number of different classes
        self.encoder = LabelEncoder()
        self.encoder.fit(y)
        self.encoded_labels = self.encoder.transform(y)

        # initiate the parameters and the minimum distances
        self.params = np.zeros((len(self.encoder.classes_), 3))
        self.distances = defaultdict(list)

        # search for the minimum distance to points of the same class
        for index, data in enumerate(self.input_data):
            label = self.encoded_labels[index] #y index
            # print(self.encoder.inverse_transform([label]))
            try:
                if len(self.input_data[self.encoded_labels == label]) == 2:
                    raise ValueError
                nn_search = LinearNNSearch(data, fit=True)
                closest = nn_search.nn_distance(self.input_data[self.encoded_labels == label]) #point, label
                # print(f'closest is: {closest}')
                self.distances[label].append(closest)
            except Exception as e:
                print(f"class: {self.encoder.inverse_transform([label])} only has one value, as such single point will not be used in classification")

        for key in self.distances.keys():
            # take the log of the minimum distances
            self.distances[key] = np.log(np.asarray(self.distances[key]))


            # move data points into support of gamma distribution and save the movement
            minimum = np.min(self.distances[key])
            self.params[key][2] = minimum
            self.distances[key] -= minimum
            self.distances[key] += self.ε

            # find the parameters for gamma distribution and save them
            self.params[key][0], self.params[key][1] = gamma_mle(self.distances[key])

    def predict(self,X, p_value = False):
        """
        X: data points for classification
        full: whether or not to give all the p-scores
        returns: np array of predictions or np array of np arrays of p-scores
        """

        # put in a place for models to be able to saved
        models = [i for i in set(self.encoded_labels)]

        # each data point we want to predict on needs an array that is of the same length as the
        # number of classes they can be classified into
        predictions = [models.copy() for x in X]

        # for each class test each point
        for each_class in set(self.encoded_labels):
            # save the models
            models[each_class] = sp.stats.gamma(self.params[each_class][0], self.params[each_class][1])
            for index, x in enumerate(X):
                # calculate distances and shift into gamma's support
                nn_search = LinearNNSearch(x)
                dist = nn_search.nn_distance(self.input_data[self.encoded_labels == each_class])
                adj_dist = dist - self.params[each_class][2] + self.ε

                # calculate the inverse cdf
                # if the adjusted isn't above zero, it is closer to a member of said class than
                # any other point
                predictions[index][each_class] = 1-models[each_class].cdf(adj_dist) if adj_dist > 0 else 1

        predictions = np.asarray(predictions)
        if p_value:
            return predictions
        prediction = np.argmax(predictions, axis = 1)
        '''
        TODO: parallized this following code:

        '''
        for index, individual in enumerate(predictions):
            if individual[prediction[index]] < self.threshold:
                prediction[index] = -1


        return prediction

    def get_classes(self):
        return self.encoder.classes_ if self.encoder != None else None

    def get_distances(self, classes = "none"):
        return self.distances if classes == 'none' else self.distances[classes]

    def get_params(self):
        return self.params[:, 0:2]

    def get_encoder(self):
        return self.encoder





    #only here so that i can scroll down while coding
