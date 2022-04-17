import scipy as sp
import numpy as np

import classifier

from matplotlib import pyplot as plt

def get_raw_p(array):
    return np.argsort(array)/(len(array) + 1)

def get_emp_p(array, k: float, theta: float):
    dist = sp.stats.gamma(theta, scale = k)
    return dist.cdf(array)

def gen_gamma_pdf(x, a, d, p):
    return (p/(a**d))* (x **(d-1)) * np.exp( -( (x/a)**p)) / special.gamma(d/p)

def graph(model, data = ''):
    params = model.get_params()
    distances = model.get_distances()

    # TODO: make subplots

    for c, dists in distances.items():
        dist_params = params[c]

        dists = np.sort(dists)
        empirical_pdf = dists / dists.size
        gamma_cdf = 1 - get_emp_p(dists, dist_params[0], dist_params[1])

        fig = plt.figure()
        ax = fig.add_subplot(221)
        plt.title(f'{data} Class {c} Size: {dists.size}')

        plt.plot(dists, gamma_cdf, label='Gamma CDF')
        plt.plot(dists, 1 - empirical_pdf.cumsum(), label='Empirical CDF')
        plt.yscale('log')
        plt.legend()

        ax1 = fig.add_subplot(222)
        ax1.plot(gamma_cdf, 1 - empirical_pdf.cumsum(), label = "Comparison")
        ax1.plot([0,1],[0,1], label = "y = x")
        ax1.set_yscale('log')
        ax1.set_xscale('log')

        ax3 = fig.add_subplot(223)
        n, bins, patches = ax3.hist(dists)
        ax3.set_yscale('linear')
        ax3.set_xscale('linear')
        x = np.linspace(0, bins[-1], 200)
        ax3.plot(x, sp.stats.gamma.pdf(x, dist_params[0], scale = dist_params[1])*dists.size)


        plt.savefig(f'graphs/{data} class {c}.png')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # # this code may or not be mostly stolen from `testing.py`
    # data= np.genfromtxt('bezdekIris.data', delimiter=",")
    #
    # #add preprocessing function
    # x = data[:, 1:-1]
    # y = data[:, -1]

    data= np.genfromtxt('bezdekIris.data', delimiter=",")
    data1 = np.genfromtxt('bezdekIris.data', delimiter=",", dtype = "S")

    #add preprocessing function
    x = np.asarray([input[:-1] for input in data])
    y = np.asarray([input[-1] for input in data1])

    model = classifier.NNClassifier()
    model.fit(x, y)
    graph(model)
