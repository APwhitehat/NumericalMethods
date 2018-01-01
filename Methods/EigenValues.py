from __future__ import division
from math import *
import numpy as np
from LinearSolve import LinearSolver


class Eigenvalue():
    methods = ['power', 'inv_power', 'inv_power_shift', 'QR']

    def __init__(self, a, max_itr, max_err, shift=0):
        self.a = a
        self.max_itr = max_itr
        self.max_err = max_err
        self.shift = shift
        self.v = None
        self.itr = 0
        self.eigens = np.array([])

    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    @staticmethod
    def get_error(v, u):
        return np.linalg.norm(v - u)

    @staticmethod
    def QR_decompose(a):
        q, r = np.zeros(shape=a.shape), np.zeros(shape=a.shape)

        for i in range(a[0].size):
            q[..., i] = a[..., i] - np.sum(r[:i, i] * q[..., :i], axis=1)
            q[..., i] = Eigenvalue.normalize(q[..., i])
            r[i, i:] = q[..., i].transpose().dot(a[..., i:])

        return q, r

    def __power_method__(self, next):
        self.v = np.ones(self.a[0].size) / sqrt(self.a[0].size)
        err = float("inf")
        while self.itr < self.max_itr and err > self.max_err:
            self.itr += 1
            self.v = next(self.v)
            # add current eigen value prediction
            self.eigens = np.append(self.eigens, np.linalg.norm(self.v))
            self.v = Eigenvalue.normalize(self.v)
            if self.eigens.size > 1:
                err = Eigenvalue.get_error(*self.eigens[-2:])

    def power(self):
        def next(v):
            return self.a.dot(v)
        self.__power_method__(next)

    def inv_power(self):
        n = self.a[0].size
        l, u = np.identity(n), np.identity(n)
        LinearSolver.LU(self.a, l, u)

        def next(v):
            return LinearSolver.solveLU(l, u, v)
        self.__power_method__(next)

        self.eigens **= -1

    def inv_power_shift(self):
        self.a = self.a - self.shift * np.identity(self.a[0].size)
        self.inv_power()
        self.eigens += self.shift

    def QR(self):
        err = float("inf")
        while self.itr < self.max_itr and err > self.max_err:
            self.itr += 1
            u = self.a
            q, r = Eigenvalue.QR_decompose(self.a)
            self.a = r.dot(q)
            # add current eigen value prediction
            self.eigens = np.append(self.eigens, self.a.diagonal())
            err = Eigenvalue.get_error(self.a.diagonal(), u.diagonal())

        n = self.a.diagonal().size
        self.eigens.shape = (self.eigens.size // n, n)


def main():
    method_names = ['Power method', 'Inverse power method',
                    'Inverse power method with shift', 'QR method']

    for i, name in enumerate(method_names):
        print 'press ', i, 'for ', name
    choice = int(raw_input())

    with open("q2.txt", "r") as f:
        n = int(f.readline())
        a = np.array([map(float, f.readline().split()) for i in range(n)])
        max_itr = int(f.readline())
        max_err = float(f.readline()) / 100
        shift = float(f.readline()) if choice == 2 else 0

    eigen_find = Eigenvalue(a, max_itr, max_err, shift)
    getattr(eigen_find, Eigenvalue.methods[choice])()

    with open("out.txt", "w") as fout:
        if eigen_find.eigens[-1] is not None:
            np.savetxt(fout, eigen_find.eigens[-1:], fmt='%f',
                       header='Eigenvalue', footer='\n', comments='')

        if eigen_find.v is not None:
            np.savetxt(fout, eigen_find.v, fmt='%f',
                       header='Eigenvector', footer='\n', comments='')

        fout.write('Iterations\n%s\n\n' % eigen_find.itr)
        np.savetxt(fout, eigen_find.eigens, fmt='%f',
                   header='Eigenvalue Estimates', comments='')


if __name__ == '__main__':
    main()
