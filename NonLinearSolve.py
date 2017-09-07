from __future__ import division
from math import *
from cmath import *
import numpy as np
import matplotlib.pyplot as plt


def sgn(number):
    # only for real part
    return number.real > 0


class NonLinearSolve():
    def __init__(self, user_func, start, max_itr, rel_err, method):
        self.user_func = user_func
        self.start = start
        self.max_itr = max_itr
        self.rel_err = rel_err
        self.error_list = []
        self.itr_list = []
        self.root = None
        self.solvemethod = method

    # Methods supported
    methods = ['Bisection', 'False_Position', 'Fixed_Point',
               'Newton_Raphson', 'Secant', 'Muller']

    # make a list of safe functions
    safe_list = ['math', 'acos', 'asin', 'atan', 'atan2', 'ceil', 'cos', 'cosh',
                 'degrees', 'e', 'exp', 'fabs', 'floor', 'fmod', 'frexp', 'hypot',
                 'ldexp', 'log', 'log10', 'modf', 'pi', 'pow', 'radians', 'sin',
                 'sinh', 'sqrt', 'tan', 'tanh', 'abs']
    safe_dict = dict([(k, globals().get(k, None)) for k in safe_list])

    @staticmethod
    def __replace_carat__(func):
        return func.replace('^', '**')

    def f(self, x, func=None):
        if func is None:
            func = self.user_func
        self.safe_dict['x'] = x
        return eval(func, {"__builtins__": None}, self.safe_dict)

    @staticmethod
    def get_error(x, l):
        if x != 0.0:
            return abs((x - l) / x)
        else:
            return abs(x - l)

    def add_to_lists(self, itr, error):
        self.itr_list.append(itr)
        self.error_list.append(error)

    def __bracketing__(self, next):
        l, r = self.start
        error = float("inf")
        itr = 0
        prev_x = l
        while error > self.rel_err and itr < self.max_itr:
            x = next(l, r)
            error = NonLinearSolve.get_error(x, prev_x)
            if sgn(self.f(l)) == sgn(self.f(x)):
                l = x
            else:
                r = x
            # Add data into lists
            self.add_to_lists(itr, error)
            prev_x = x
            self.root = x
            itr += 1

    def __iterative__(self, next):
        x_list = [i for i in self.start]
        error = float("inf")
        itr = 0
        while error > self.rel_err and itr < self.max_itr:
            x = next(x_list)
            error = NonLinearSolve.get_error(x, x_list[-1])
            # Add data into lists
            self.add_to_lists(itr, error)
            self.root = x
            x_list.append(x)
            itr += 1

    def Bisection(self):
        self.__bracketing__(lambda x, y: (x + y) / 2)

    def False_Position(self):
        def next(l, r):
            return l - (r - l) * self.f(l) / (self.f(r) - self.f(l))
        self.__bracketing__(next)

    def Fixed_Point(self):
        phi_func = raw_input('Enter the function phi(x):')

        def next(x_list):
            return self.f(x_list[-1], phi_func)
        self.__iterative__(next)

    def Newton_Raphson(self):
        f_dash = raw_input("Enter the function f'(x):")

        def next(x_list):
            x = x_list[-1]
            return x - self.f(x) / self.f(x, f_dash)
        self.__iterative__(next)

    def Secant(self):
        def next(x_list):
            y, x = x_list[-1], x_list[-2]
            return y - (y - x) * (self.f(y)) / (self.f(y) - self.f(x))
        self.__iterative__(next)

    def Muller(self):
        def next(x_list):
            x = x_list
            y = [self.f(x[i]) for i in (-3, -2, -1)]
            c = y[-1]
            a = (y[-1] - y[-3]) / (x[-1] - x[-3]) - \
                (y[-2] - y[-1]) / (x[-2] - x[-1])
            a /= (x[-3] - x[-2])
            b = (y[-1] - y[-2]) / (x[-1] - x[-2]) * (x[-3] - x[-1]) - \
                (x[-2] - x[-1]) * (y[-3] - y[-1]) / (x[-3] - x[-1])
            b /= (x[-3] - x[-2])
            det = sqrt(abs(b)**4 - 4 * a * c * (b.conjugate()**2))
            delx = (-2 * c * b.conjugate()) / (abs(b)**2 + det)
            return x[-1] + delx
        self.__iterative__(next)

    @staticmethod
    def __format__(root):
        if root.imag == 0:
            return root.real
        return root

    def get_root(self):
        return self.__format__(self.root)

    def __plot_fx__(self, xvals):
        yvals = list(self.f(i) for i in xvals)
        plt.plot(xvals, yvals)
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('f(x) vs x')
        plt.show()

    def plot_fx(self):
        xvals = np.arange(self.root.real - 10, self.root.real + 10, 0.01)
        self.__plot_fx__(xvals)

    def plot_error(self):
        plt.plot(self.itr_list, self.error_list)
        plt.grid()
        plt.xlabel('Iteration no.')
        plt.ylabel('Error')
        plt.title('Relative approximate error vs iteration number')
        plt.show()


class Bairstow(NonLinearSolve):
    '''
    This method finds all roots of a polynimial
    This method supports polynomial only.
    '''
    methods = NonLinearSolve.methods + ['Bairstow']

    def __init__(self, user_func, start, max_itr, rel_err, method):
        NonLinearSolve.__init__(self, user_func, start,
                                max_itr, rel_err, method)
        self.roots_left = len(self.user_func) - 1
        self.poly = self.user_func

    @staticmethod
    def get_error(delx, x):
        if x != 0.0:
            return delx / x
        else:
            return delx

    def f(self, x):
        fx, term = 0, 1
        for coef in self.user_func[::-1]:
            fx += term * coef
            term *= x
        return fx

    def find(self):
        if len(self.poly) < 3:
            if len(self.poly) == 2:
                self.root = [self.poly[1] / self.poly[0]]
            self.poly = []
            self.roots_left = 0
            return
        a = self.start
        error = float("inf")
        itr = 0
        while error > self.rel_err and itr < self.max_itr:
            d = [self.poly[0], self.poly[1] + a[1] * self.poly[0]]
            for i in range(len(self.poly) - 2):
                d.append(self.poly[i + 2] + a[1] * d[i + 1] + a[0] * d[i])
            del_d = [0, d[0], d[1] + a[1] * d[0]]
            for i in range(1, len(self.poly) - 2):
                del_d.append(d[i + 1] + a[1] * del_d[i + 1] + a[0] * del_d[i])
            matx = np.array([[del_d[-2], del_d[-1]],
                             [del_d[-3], del_d[-2]]])
            vecb = np.array([-d[-1], -d[-2]])
            del_a = np.linalg.solve(matx, vecb)
            error = max(Bairstow.get_error(
                del_a[0], a[0]), Bairstow.get_error(del_a[1], a[1]))
            a[0] += del_a[0]
            a[1] += del_a[1]
            itr += 1
        else:
            self.poly = d[:-2]
        det = sqrt(a[1]**2 + 4 * a[0])
        self.root = [0.5 * (a[0] + det), 0.5 * (a[1] - det)]
        self.roots_left = len(self.poly) - 1

    def update_start(self, start):
        self.start = start

    def get_root(self):
        roots = [str(NonLinearSolve.__format__(r)) for r in self.root]
        return ', '.join(roots)

    def get_roots_left(self):
        return self.roots_left

    def plot_fx(self):
        xvals = np.arange(self.root[0].real - 10, self.root[0].real + 10, 0.01)
        self.__plot_fx__(xvals)


def main():

    print 'Press -1 to quit'
    for i, method in enumerate(Bairstow.methods):
        print 'Press ', i, 'for', method, 'Method'
    opt = int(raw_input())
    if opt == -1:
        return

    if Bairstow.methods[opt] == 'Bairstow':
        user_func = map(float, raw_input(
            "Enter the polynomial coefficients:\n%s " %
            '[starting from nth degree to constant, separated by spaces]:'
        ).split())
    else:
        user_func = raw_input('Enter the function f(x): ')
        user_func = NonLinearSolve.__replace_carat__(user_func)
    start = map(float, raw_input(
        'Enter starting values(comma separated): ').split(','))
    max_itr = int(raw_input('Enter the maximum no. of iterations: '))
    rel_err = float(
        raw_input('Enter the required relative approximate error %: ')) / 100.0

    if Bairstow.methods[opt] == 'Bairstow':
        solver = Bairstow(user_func, start, max_itr, rel_err,
                          Bairstow.methods[opt])
        while True:
            solver.find()
            print 'Roots are: ', solver.get_root()
            if solver.get_roots_left() > 0:
                print 'Number of roots left: ', solver.get_roots_left()
                if solver.get_roots_left() > 1:
                    start = map(float, raw_input(
                        'Enter starting values(comma separated): ').split(','))
                solver.update_start(start)
            else:
                break
        if raw_input('Plot f(x) vs x?[Press 1 for yes]: ') is '1':
            solver.plot_fx()
    else:
        solver = NonLinearSolve(user_func, start, max_itr, rel_err,
                                NonLinearSolve.methods[opt])
        getattr(solver, NonLinearSolve.methods[opt])()

        print 'Root of f(x)=0 is: ', solver.get_root()
        if raw_input('Plot f(x) vs x?[Press 1 for yes]: ') is '1':
            solver.plot_fx()
        if raw_input("Plot relative approximate error vs iteration number?%s: " %
                     '[Press 1 for yes]') is '1':
            solver.plot_error()


if __name__ == '__main__':
    main()
