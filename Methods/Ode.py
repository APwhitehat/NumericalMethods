from __future__ import division
import numpy as np
from utils import Grapher, SafeEval


class Ode():
    methods = ['euler', 'rk2', 'rk4']

    def __init__(self, func, a, b, y0, h):
        self.f = SafeEval.functionize(func)
        self.a = a
        self.b = b
        self.h = h
        self.x = [a]
        self.y = [y0]
        self.integral = None

    def intergrate(self, next):
        eps = self.h / 100
        while self.x[-1] + eps < self.b:
            self.y.append(next(self.x[-1], self.y[-1]))
            self.x.append(self.x[-1] + self.h)

        self.integral = self.y[-1]

    def euler(self):
        def next(x, y):
            return y + self.h * self.f(x, y)
        self.intergrate(next)

    def rk2(self):
        def next(x, y):
            x_, y_ = x + self.h / 2, y + self.h / 2 * self.f(x, y)
            return y + self.h * self.f(x_, y_)
        self.intergrate(next)

    def rk4(self):
        def next(x, y):
            h, f = self.h, self.f
            k1 = f(x, y)
            k2 = f(x + h / 2, y + h / 2 * k1)
            k3 = f(x + h / 2, y + h / 2 * k2)
            k4 = f(x + h, y + h * k3)
            return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.intergrate(next)

    def plot(self):
        Grapher.plot(x=self.x, y=self.y)


def main():

    with open("q2.txt", "r") as f:
        func = f.readline().replace('t', 'x')
        a, y0 = map(float, f.readline().split(', '))
        b = float(f.readline())
        h = float(f.readline())
        op = int(f.readline())

    ode = Ode(func, a, b, y0, h)
    getattr(ode, Ode.methods[op - 1])()

    with open("out.txt", "w") as fout:
        if ode.integral is not None:
            np.savetxt(fout, np.array([ode.x, ode.y]).transpose(),
                       fmt='%f', header='t,       y', comments='')
    if raw_input('Plot f(x) vs x?[Press 1 for yes]: ') is '1':
        ode.plot()


if __name__ == '__main__':
    main()
