from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from LinearSolve import LinearSolver


class Grapher:
    ''' plot f in range l, r '''
    @staticmethod
    def plot(f, l, r, x=None, y=None, xlabel='x', ylabel='f(x)', title='f(x) vs x', points=1000):
        if x is not None and y is not None:
            plt.scatter(x, y)
        xvals = np.arange(l, r, (r - l) / points)
        yvals = map(f, xvals)
        plt.plot(xvals, yvals)
        plt.grid()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()


class Interpolator():
    methods = ['lagrange', 'spline']

    def __init__(self, x, y, xstar):
        self.x = x
        self.y = y
        self.xstar = np.sort(xstar)
        self.ystar = None
        self.N = x.size
        self.s = np.array([0] * self.N)
        self.f = None

    ''' Returns i'th Lagrange polynomial evaluated at pt '''
    @staticmethod
    def L(i, x, pt):
        trans = map(lambda arg: (pt - arg) / (x[i] - arg), x[np.arange(len(x)) != i])
        return np.prod(trans)

    def lagrange(self):
        L = Interpolator.L

        def func(pt):
            return np.sum([L(i, self.x, pt) * self.y[i] for i in range(self.N)])
        self.evaluate(func)
        self.f = func

    def spline(self):
        n, x, y, s = self.N, self.x, self.y, self.s
        a, b = np.ndarray(shape=(n - 2, n)), np.ndarray(shape=(n - 2))

        ''' Divided Diference '''
        def dd(i, j):
            return (y[i] - y[j]) / (x[i] - x[j])

        for i in range(1, n - 1):
            a[i - 1, i - 1:i + 2] = [x[i] - x[i - 1], 2 *
                                     (x[i + 1] - x[i - 1]), x[i + 1] - x[i]]
            b[i - 1] = 6 * (dd(i + 1, i) - dd(i, i - 1))
        a = a[..., 1:n - 1]

        solver = LinearSolver(a, b)
        solver.GE_pivoted()
        s[1:-1] = solver.x

        def S(i, pt):
            return ((x[i + 1] - pt)**3 * s[i] + (pt - x[i])**3 * s[i + 1]) / 6 / (x[i + 1] - x[i]) \
                + (y[i] / (x[i + 1] - x[i]) - (x[i + 1] - x[i]) * s[i] / 6) * (x[i + 1] - pt) \
                + (y[i + 1] / (x[i + 1] - x[i]) - (x[i + 1] - x[i]) * s[i + 1] / 6) * (pt - x[i])

        def func(pt):
            for i, x in enumerate(self.x):
                if pt < x:
                    return S(i - 1, pt)
        self.evaluate(func)
        self.f = func

    def evaluate(self, f):
        self.ystar = np.array([f(i) for i in self.xstar])


def main():
    method_names = ['Lagrange polynomials', 'Natural Cubic Spline']

    for i, name in enumerate(method_names):
        print 'press ', i, 'for ', name
    choice = int(raw_input())

    with open("q1.txt", "r") as f:
        n = int(f.readline())
        a = np.array([map(float, f.readline().split()) for i in range(n)])
        x, y = a[..., 0], a[..., 1]
        m = int(f.readline())
        xstar = np.array([float(f.readline()) for i in range(m)])

    func = Interpolator(x, y, xstar)
    getattr(func, Interpolator.methods[choice])()
    Grapher.plot(func.f, func.x[0], func.x[-1], x=func.x, y=func.y)

    with open("out.txt", "w") as fout:
        if func.ystar is not None:
            formated_y = np.array([func.xstar, func.ystar]).transpose()
            np.savetxt(fout, formated_y, fmt='%f',
                       header=method_names[choice], footer='\n', comments='')


if __name__ == '__main__':
    main()
