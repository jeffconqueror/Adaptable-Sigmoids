import numpy as np
from distribution_functions import GenGamma


def bracket_minimum(f, x=0, s=2e-9, k=1.1):
    a, ya = x, f(x)
    b, yb = a + s, f(a + s)
    if yb > ya:
        a, b = b, a
        ya, yb = yb, ya
        print("yb > ya")
        s = -s
    while 1:
        c, yc = b + s, f(b + s)
        if yc > yb:
            return (a, c) if a < c else (c, a)
        a, ya, b, yb = b, yb, c, yc
        s *= k

def line_search(f, x, d):
    # print(f"direction is {d}")
    def objective(z):
        return f(x + z*d)
    a, b = bracket_minimum(objective)
    z = golden_section_search_minimize(objective, a, b, .002)
    return x + z*d

def golden_section_search_minimize(f, a, b, min, n = 10000):
    ϕ = (1 + 5 ** 0.5) / 2
    ρ = ϕ-1
    d = ρ * b + (1 - ρ)*a
    yd = f(d)
    count = 0
    while abs(a - b) > min and count < n:
        c = ρ*a + (1 - ρ)*b
        yc = f(c)
        if yc < yd:
            b, d, yd = d, c, yc
        else:
            a, b = b, c
        count = count + 1
    return (a+b)/2

class BFGS:
    def __init__(self, f, del_f, x, threshold = 0.0001, max_iters = 10000):
        m = len(x)
        self.Q = np.identity(m)
        self.f = f
        self.del_f = del_f
        self.threshold = threshold
        self.max = max_iters

    def __init__(self, threshold = .0001, max_iters = 10000):
        self.threshold = threshold
        self.max = max_iters

    def step(self, x):
        Q, g = self.Q, self.del_f(x)
        # print(f"g is {g}")
        x_prime = line_search(self.f, x, np.dot(-Q,g))
        g_prime = self.del_f(x_prime)
        δ = x_prime - x
        γ = g_prime - g
        # print(f"δ.T*γ is {δ.T}*{γ}")
        Q = Q - (δ*γ.T*Q + Q*γ*δ.T)/(δ.T*γ) + (1 + (γ.T*Q*γ)/(δ.T*γ))[1]*(δ*δ.T)/(δ.T*γ)
        return x_prime

    def aprox(self, x):
        print("approximating")
        old = x
        x = self.step(x)
        i = 0
        while np.mean(np.abs(old - x)) > self.threshold:
            if i == self.max:
                break
            old = x
            x = self.step(x)
            i += 1
        return x

class frechet_BFGS(BFGS):
    def __init__(self, distribution, threshold = 1e-7, max_iters = 10000):
        m = 3
        self.Q = np.identity(m)
        def log_likely(params):
            return neg_frechet_log_likely(distribution, params)

        def Δlog_likely(params):
            return neg_Δfrechet_log(distribution, params)

        self.f = log_likely
        self.del_f = Δlog_likely
        self.threshold = threshold
        self.max = max_iters

class gen_gamma_BFGS(BFGS):
    def __init__(self, distribution):
        gamma = GenGamma(distribution)
        self.Q = np.identity(3)
        def ab(x):
            return -gamma.log_likely(x)
        self.f = ab
        self.del_f = lambda x: -gamma.Δlog_likely(x)
        super().__init__()

### test functions
def quad(x, vals = [.1,.2,.3,.4,.5,.6,.232,.9,.98,.0012]):
    sum = 0
    for i in range(len(vals)):

        sum += (x[i]-vals[i])**2
    return sum

def del_quad(x, vals = [.1,.2,.3,.4,.5,.6,.232,.9,.98,.0012]):
    del_q = []
    for i in range(len(vals)):
        del_q.append(2*(x[i] - vals[i]))
    return np.asarray(del_q)

##frechet functions
# grabbed functions from https://www.hindawi.com/journals/cmmm/2019/9089856/

def frechet_log_likely(data, params = np.asarray([1,1,1])):
    n = len(data)
    α = params[0]
    λ = params[1]
    η = params[2]
    # print(f"λ is {λ}")
    likelihood = n*np.log(α) + n*α*np.log(λ) - (α + 1)*np.sum(np.log(np.abs(data -η))) - np.sum( ( (data-η) / λ )** (-α))
    return likelihood

def neg_frechet_log_likely(data, params = np.asarray([1,1,1])):
    return -frechet_log_likely(data, params)

def Δfrechet_log(data, params = np.asarray([1,1,1])):
    n = len(data)
    α = params[0]
    λ = params[1]
    η = params[2]

    ## added γ to make code more readable

    γ = (((data - η) / λ) ** (-α)) * np.log( (data - η)/λ)

    Δα =  n/α + n*np.log(λ) - np.sum(np.log(data - η)) + np.sum(γ)
    Δλ = (n*α/λ) - (α/λ) * np.sum( ( (data - η)/λ ) ** (-α) )
    Δη = (α + 1) *np.sum( (data - η) ** (-1)) - (α/λ) * np.sum( ( (data - η)/λ ) ** (-α-1) )
    return np.asarray([Δα, Δλ, Δη])

def neg_Δfrechet_log(data, params = np.asarray([1,1,1])):
    return -Δfrechet_log(data, params)


from scipy.stats import invweibull
import matplotlib.pyplot as plt

if __name__ == "__main__":
    c = 2
    mean, var, skew, kurt = invweibull.stats(c, moments='mvsk')

    r = invweibull.rvs(c, size=10000)

    test_opt = frechet_BFGS(r)
    print(test_opt.aprox(np.asarray([1,1,0])))
