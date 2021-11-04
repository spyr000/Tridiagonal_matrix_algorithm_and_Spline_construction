import numpy as np
import pandas as pd

A = np.array([
    [1, 4, 0, 0, 0],
    [1, 2, 3, 0, 0],
    [0, 1, 2, 3, 0],
    [0, 0, 1, 2, 3],
    [0, 0, 0, 5, 2]
])
B = np.array([1, 2, 3, 4, 5])
# A = np.array([
#     [2,-1,0,0],
#     [4,2,1,0],
#     [0,4,5,2],
#     [0,0,3,-1]
# ])
#
# B = [2,8,10,1]

assert A.shape[0] == A.shape[1]
M_DIMENSION = A.shape[0]

def tridiag(A, B):
    b = pd.Series(B, index=pd.RangeIndex(1, M_DIMENSION + 1))
    c = pd.Series(np.diagonal(A, offset=-1), index=pd.RangeIndex(2, M_DIMENSION + 1))
    d = pd.Series(np.diagonal(A), index=pd.RangeIndex(1, M_DIMENSION + 1))
    e = pd.Series(np.diagonal(A, offset=1), index=pd.RangeIndex(1, M_DIMENSION))

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

res = tridiag(A,B)

print("Вектор Х = ", res)
print(all(np.isclose(A.dot(res), B)))
