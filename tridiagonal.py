import numpy as np
import pandas as pd

def tridiag(b,c,d,e):

    M_DIMENSION = b.shape[0]


    b = pd.Series(b, index=pd.RangeIndex(1, M_DIMENSION + 1))
    c = pd.Series(c, index=pd.RangeIndex(2, M_DIMENSION + 1))
    d = pd.Series(d, index=pd.RangeIndex(1, M_DIMENSION + 1))
    e = pd.Series(e, index=pd.RangeIndex(1, M_DIMENSION))

    alphas, betas = pd.Series(-e[1] / d[1], index=[2]), pd.Series(b[1] / d[1], index=[2])

    #straight run
    for i in range(3, M_DIMENSION + 1):
        denom = d[i - 1] + c[i - 1] * alphas[i - 1]
        # print(b[i - 1])
        alphas = alphas.append(pd.Series(-e[i - 1] / denom, index=[i]))
        betas = betas.append(pd.Series((-c[i - 1] * betas[i - 1] + b[i - 1]) / denom, index=[i]))

    # reversed run
    x = pd.Series(np.zeros(M_DIMENSION), index=pd.RangeIndex(1, M_DIMENSION+1))
    x[M_DIMENSION] = (-c[M_DIMENSION] * betas[M_DIMENSION] + b[M_DIMENSION]) / \
                     (d[M_DIMENSION] + c[M_DIMENSION] * alphas[M_DIMENSION])

    for i in range(M_DIMENSION - 1, 0, -1):
        x[i] = alphas[i + 1] * x[i + 1] + betas[i + 1]

    return x.values