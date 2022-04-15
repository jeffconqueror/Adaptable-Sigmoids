import numpy as np
import classifier
import sklearn.model_selection
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)


def test(X, y):
    xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(X,y)

    model = classifier.NNClassifier()
    model.fit(xTrain, yTrain)

    predictions = model.predict(xTest)
    print(predictions)
    encoder = model.get_encoder()
    yTest = encoder.transform(yTest)
    print(yTest)
    print(predictions == yTest, sum(predictions == yTest), len(yTest))


data= np.genfromtxt('bezdekIris.data', delimiter=",")
data1 = np.genfromtxt('bezdekIris.data', delimiter=",", dtype = "S")

#add preprocessing function
x = np.asarray([input[:-1] for input in data])
y = np.asarray([input[-1] for input in data1])

test(x,y)
