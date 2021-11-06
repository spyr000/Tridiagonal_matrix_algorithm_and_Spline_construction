import numpy as np
import pandas as pd
from tridiagonal import tridiag
import matplotlib.pyplot as plt

def generate_spline(x, y):

    h = abs(x[-1] - x[0]) / (len(x) - 1)
    y = np.array(y)

    # 1
    phi_i = (6 / h) * (y[2:] - 2 * y[1:-1] + y[:-2])

    # 2
    dimension = phi_i.shape[0]
    c = tridiag(phi_i, np.full(dimension - 1, h), np.full(dimension, 4 * h), np.full(dimension - 1, h))
    c = np.append(c, np.array([0]))

    # 3
    d = (c[1:] - c[:-1]) / h
    d = np.append([c[0] / h], d)

    # 4
    b = (y[1:] - y[:-1]) / h + (h / 2) * c - (h * h / 6) * d

    # 5+6
    def S(xx):
        arr_xx = np.array(xx)
        arr_x = np.array(x)
        segments = []

        for i in range(dimension + 1):
            segments.append(arr_xx[np.logical_and(arr_x[i] <= arr_xx, arr_xx < arr_x[i + 1])])

        i = 0
        result = []
        for arr in segments:
            for arg in arr:
                result.append(
                    y[i + 1] + \
                    b[i] * (arg - x[i + 1]) + \
                    (arg - x[i + 1]) * (arg - x[i + 1]) * (c[i] / 2) + \
                    (arg - x[i + 1]) * (arg - x[i + 1]) * (arg - x[i + 1]) * (d[i] / 6)
                )
            i += 1
        return result

    return S

a = 0
b = 3
res = []
xx = np.arange(a, b, 0.01)

f = lambda x: x ** (x * np.cos(x))
func = np.vectorize(f)

fig, ax = plt.subplots()
for n in [5, 10, 20]:
    x = np.linspace(a, b, n)
    y = f(x)
    spline = generate_spline(x, y)
    res.append(spline(xx))

plt.plot(xx, func(xx), 'b--', xx, res[0], 'r', xx, res[1], 'g', xx, res[2], 'y')
plt.show()
plt.show()

