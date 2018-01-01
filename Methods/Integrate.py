from __future__ import division
import numpy as np
from utils import Grapher, memoize, SafeEval


class Romberg():
    def __init__(self, func, a, b, err):
        self.f = SafeEval.functionize(func)
        self.a = a
        self.b = b
        self.err = err
        self.intervals = None
        self.integral = None
        self.final_err = None

    @staticmethod
    def trapezoidal(f, l, r, n):
        h = (r - l) / n
        return h * ((f(l) + f(r)) / 2 + sum(map(f, np.arange(l + h, r, h))))

    def integrate(self):
        @memoize
        def T(n):
            return Romberg.trapezoidal(self.f, self.a, self.b, n)

        @memoize
        def R(o, n):
            if o == 2:
                return T(n)
            else:
                return (4 * R(o - 2, 2 * n) - R(o - 2, n)) / 3

        def err(o):
            e = R(o, 1) - R(o - 2, 2)
            if R(o, 1) is not 0:
                e /= R(o, 1)
            return e
        o = 4
        while abs(err(o)) > self.err:
            o *= 2
        self.integral = R(o, 1)
        self.intervals = o / 2
        self.final_err = err(o)

    def plot(self):
        x = np.linspace(self.a, self.b, self.intervals + 1)
        y = [self.f(_) for _ in x]
        Grapher.plot(x=x, y=y)


class Quadrature():
    def __init__(self, func, a, b, err):
        self.f = SafeEval.functionize(func)
        self.a = a
        self.b = b
        self.err = err
        self.points = None
        self.integral = None
        self.final_err = None

    @staticmethod
    def gaussxw(N):
        # Initial approximation to roots of the Legendre polynomial
        a = np.linspace(3, 4 * N - 1, N) / (4 * N + 2)
        x = np.cos(np.pi * a + 1 / (8 * N * N * np.tan(a)))

        # Find roots using Newton's method
        epsilon, delta = 1e-15, 1.0
        while delta > epsilon:
            p0 = np.ones(N, float)
            p1 = np.copy(x)
            for k in range(1, N):
                p0, p1 = p1, ((2 * k + 1) * x * p1 - k * p0) / (k + 1)
            dp = (N + 1) * (p0 - x * p1) / (1 - x * x)
            dx = p1 / dp
            x -= dx
            delta = np.max(abs(dx))

        # Calculate the weights
        w = 2 * (N + 1) * (N + 1) / (N * N * (1 - x * x) * dp * dp)

        return x, w

    @staticmethod
    def gaussxwab(N, a, b):
        x, w = Quadrature.gaussxw(N)
        return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w

    def intergrate(self):
        @memoize
        def G(n):
            x, w = Quadrature.gaussxwab(n, self.a, self.b)
            fx = np.array([self.f(i) for i in x])
            w = np.array(w)
            return np.sum(fx * w)

        def err(n):
            e = G(n) - G(n - 1)
            if G(n) is not 0:
                e /= G(n)
            return e
        n = 2
        while abs(err(n)) > self.err:
            n += 1
        self.points = n
        self.integral = G(n)
        self.final_err = err(n)

    def plot(self):
        x, w = Quadrature.gaussxwab(self.points, self.a, self.b)
        y = [self.f(_) for _ in x]
        Grapher.plot(x=x, y=y)


def main():

    with open("q1.txt", "r") as f:
        func = f.readline()
        a, b = map(float, f.readline().split(', '))
        err = float(f.readline())
        op = int(f.readline())

    if op == 1:
        romberg = Romberg(func, a, b, err / 100)
        romberg.integrate()
        with open("out.txt", "w") as fout:
            if romberg.integral is not None:
                fout.write('I= %f\n' % romberg.integral)
                fout.write('Number of intervals= %d\n' % romberg.intervals)
                fout.write('Approximate relative error (%%)= %f\n' %
                           (romberg.final_err * 100))
        if raw_input('Plot f(x) vs x?[Press 1 for yes]: ') is '1':
            romberg.plot()
    else:
        gauss = Quadrature(func, a, b, err / 100)
        gauss.intergrate()
        with open("out.txt", "w") as fout:
            if gauss.integral is not None:
                fout.write('I= %f\n' % gauss.integral)
                fout.write('Number of points= %d\n' % gauss.points)
                fout.write('Approximate relative error (%%)= %f\n' %
                           (gauss.final_err * 100))
        if raw_input('Plot f(x) vs x?[Press 1 for yes]: ') is '1':
            gauss.plot()


if __name__ == '__main__':
    main()
