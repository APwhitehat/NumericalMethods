from __future__ import division
import numpy as np
from LinearSolve import LinearSolver
from utils import Grapher


class Regressor():
    def __init__(self, x, y, dg):
        self.x = x
        self.y = y
        self.N = x.size
        self.dg = dg
        self.c = None
        self.cod = None
        self.f = None

    def regress(self):
        m = self.dg + 1
        sigma_x = np.array([np.sum(self.x**i) for i in range(m * 2 + 1)])
        print sigma_x
        a, b = np.ndarray(shape=(m, m)), np.ndarray(shape=(m))
        for i in range(m):
            a[i] = sigma_x[i:i + m]
            b[i] = np.sum((self.x**i) * self.y)

        solver = LinearSolver(a, b)
        solver.GE_pivoted()
        self.c = solver.x

        def func(pt):
            return np.sum(self.c * np.array([pt**i for i in range(m)]))
        self.f = func

        ''' coefficient of determination '''
        y_mean = np.mean(self.y)
        st = np.sum((self.y - y_mean)**2)
        sr = np.sum((self.y - map(func, self.x))**2)
        self.cod = (st - sr) / st


def main():

    degree = int(raw_input('Enter the degree of Polynomial:'))

    with open("q2.txt", "r") as f:
        n = int(f.readline())
        a = np.array([map(float, f.readline().split()) for i in range(n)])
        x, y = a[..., 0], a[..., 1]

    func = Regressor(x, y, degree)
    func.regress()
    Grapher.plot(func.f, np.amin(func.x), np.amax(func.x), x=func.x, y=func.y)

    with open("out.txt", "w") as fout:
        if func.c is not None:
            np.savetxt(fout, func.c, fmt='%f', header='Coefficients of the polynomial(degree %d):\n' %
                       degree, footer='\n\n', comments='', newline=' ')
            fout.write('Coefficient of determination= %f\n' % func.cod)


if __name__ == '__main__':
    main()
