from __future__ import division
from math import *
import numpy as np
import matplotlib.pyplot as plt


class memoize(dict):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        return self[args]

    def __missing__(self, key):
        result = self[key] = self.func(*key)
        return result


class Grapher:
    ''' plot f in range l, r '''
    @staticmethod
    def plot(f=None, l=None, r=None, x=None, y=None, xlabel='x', ylabel='f(x)', title='f(x) vs x', points=1000):
        if x is not None and y is not None:
            plt.scatter(x, y)
        if f is not None:
            xvals = np.linspace(l, r, points)
            yvals = map(f, xvals)
            plt.plot(xvals, yvals)
        plt.grid()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()


class SafeEval:
        # list of safe functions
    safe_list = ['math', 'acos', 'asin', 'atan', 'atan2', 'ceil', 'cos', 'cosh',
                 'degrees', 'e', 'exp', 'fabs', 'floor', 'fmod', 'frexp', 'hypot',
                 'ldexp', 'log', 'log10', 'modf', 'pi', 'pow', 'radians', 'sin',
                 'sinh', 'sqrt', 'tan', 'tanh', 'abs']
    safe_dict = dict([(k, globals().get(k, None)) for k in safe_list])

    @staticmethod
    def functionize(func, x=True, y=False):
        good_func = func.replace('^', '**')

        def f(X, Y=None):
            SafeEval.safe_dict['x'] = X
            if Y is not None:
                SafeEval.safe_dict['y'] = Y
            return eval(good_func, {"__builtins__": None}, SafeEval.safe_dict)
        return f
