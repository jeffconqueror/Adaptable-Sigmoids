### Helper functions ###
import scipy as sp
import numpy as np


def gamma_mle(data: np.ndarray, iterations: int = 4):
    '''
    data: numpy array of the data that is to be fitted to a gamma distribution
    interations: number of times the loop is to run, 4 tends to be sufficint

    returns an array of the parameters for ~Γ(shape, scale)
    '''
    #using Gamma(shape,scale) not Gamma(shape, rate)
    alpha = [0,0] # 0 is k, 1 is theta
    x = [0,0] #0 is np.log(np.mean(x)) 1 is np.mean(np.log(x))

    # print(x[0])
    x[0] = np.log(np.mean(data))
    # print(np.log(np.mean(data)))
    x[1] = np.mean(np.log(data))

    # print(np.mean(data), np.log(data))
    print(len(data))
    if x[0] == x[1]:
        print(f"error, {x[0], x[1], data}")
    alpha[0]= .5/(x[0] - x[1])

    k = alpha[0]
    for i in range(iterations):
        digamma = sp.special.digamma(k)
        digamma_prime = sp.special.polygamma(1, k)
        k = 1/ (1/k + (x[1] - x[0] + np.log(k) - digamma)/(k**2*(1/k - digamma_prime)))

    alpha[0] = k
    alpha[1] = np.exp(x[0])/alpha[0]
    return alpha
