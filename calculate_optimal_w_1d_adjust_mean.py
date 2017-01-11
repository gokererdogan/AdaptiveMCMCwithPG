"""
Learning Data-Driven Proposals through Reinforcement Learning

This script calculates the optimal (i.e., minimizes variance) w for a 1D Gaussian proposal
with state-dependent mean. Note this is only an approximation because we don't take the
accept/reject step into account.

https://github.com/gokererdogan
13 Dec. 2016
"""

"""
Here, we would like to understand what the optimal w should be for a 1D Gaussian proposal with state-dependent
mean. Let us denote the states of the chain with x_1, x_2, ..., x_T.
We want the variance of the mean=(1/T) \sum x_i to be as small as possible,
    min Var(x_1 + x_2 + x_3 + ... + x_T)
We know that
    x_1 ~ N(x_0, 1)
    x_2|x_1 ~ N((w+1)x_1 + b, 1)
    x_3|x_2 ~ N((w+1)x_2 + b, 1)
and so on.
Also,
    Var(x_1 + x_2 + ... + x_T) = Var(x_1) + Var(x_2) + ... + 2*Cov(x_1,x_2) + 2*Cov(x_1, x_3) + ...
In other words, the total variance is the sum of the elements in the covariance matrix of the joint distribution
of x_1, x_2, ..., x_T.
The joint can be calculated using the standard results for Gaussian distribution (See pg. 92-93 of Bishop).
For example, for T=2, the covariance matrix looks like
    | 1       (w+1)       |
    | (w+1)   1 + (w+1)^2 |

We can sum all the terms in this matrix, set the derivative to 0 to find the optimal w.
The code below does exactly that for different T. (Since we find the minimum for u=w+1 below, the optimal w is u-1).

NOTE: This whole derivation ignores the sampling dynamics (i.e., accept/reject step) therefore is only an approximation.
The optimal w is likely smaller because rejection increases the covariance between consecutive timesteps. Therefore,
it makes sense to move the next mean closer to the mean of the target to decrease the probability of rejection.
"""
import numpy as np
from init_plotting import *

ws = []
m_prev = np.zeros((1, 1), dtype=object)
m_prev[0, 0] = (0,)
for n in range(2, 50):
    print n
    # construct m
    m = np.zeros((n, n), dtype=object)
    m[0:(n-1), 0:(n-1)] = m_prev
    m[0:(n-1), -1] = [tuple(np.array(e)+1) for e in m_prev[-1, :]]
    m[-1, 0:(n-1)] = [tuple(np.array(e)+1) for e in m_prev[-1, :]]
    m[-1, -1] = (0,) + tuple(np.array(m_prev[-1, -1]) + 2)
    # minimize total variance
    max_exp = 2*(n-1)
    # find the coefficient of each term
    coeffs = np.zeros(max_exp+1)
    for e in range(max_exp+1):
        coeffs[e] = np.sum([e in cell for cell in np.ravel(m)])
    # take the derivative
    deriv_coeffs = (coeffs * np.arange(coeffs.size))[1:]
    # find the roots
    roots = np.roots(deriv_coeffs[::-1])
    ws.append(np.real(roots[np.imag(roots) == 0]))
    m_prev = m

ws = np.array(ws) - 1.0
plt.plot(range(2, 50), ws)
plt.xlabel("Number of timesteps")
plt.ylabel("Optimal w")
plt.savefig("optimal_w.png")
