import abc
from scipy import special
import numpy as np

digamma = special.digamma
gamma = special.gamma
log = np.log
class Distribution(abc.ABC):

    def __init__(self, data):
        self.data = np.asarray(data)
        self.length = len(data)

    @abc.abstractmethod
    def log_likely(self):
        pass

    @abc.abstractmethod
    def Δlog_likely(self):
        pass


class GenGamma(Distribution): #generalized gamma
    def log_likely(self, params):
        n = self.length
        a = params[0]
        d = params[1]
        p = params[2]
        print(f"params are {params}")
        return n*np.log(p) - n*d*log(a) - n*log(gamma(d/p)) + (d-1)*np.sum(log(self.data)) - np.sum( (self.data/a) ** p)
    #last three terms are so that a d and p are positive

    def Δlog_likely(self, params):
        n = self.length
        a = params[0]
        d = params[1]
        p = params[2]

        Δa = -n*d/a + p/a * np.sum( (self.data/a) ** p)
        Δd = -n*log(a) - n*digamma(d/p)/p + np.sum( log(self.data) )
        Δp = n/p + (n*d*digamma(d/p))/(p**2) - np.sum( (self.data**p) * log(self.data))/ (a**p) + log(a)/a * np.sum(self.data ** p)
        return np.asarray([Δa, Δd, Δp])
