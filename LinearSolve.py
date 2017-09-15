from __future__ import division
from math import *
import numpy as np


class LinearSolver():
    methods = ['GE', 'GE_pivoted', 'doolittle', 'crout', 'cholesky']

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.L = None
        self.U = None
        self.x = None

    @staticmethod
    def pivot(a, b, i):
        '''Partial Pivioting at i, i'''

        m = max(a[i:, i])
        for j in range(i + 1, a[0].size):
            if m == a[j, i]:
                a[..., i:][[i, j]] = a[..., i:][[j, i]]
                b[[i, j]] = b[[j, i]]
                break

    @staticmethod
    def solve(a, b):
        '''Expected a to be upper triangular square matrix'''

        n = b.size
        x = np.ndarray(shape=b.shape, dtype=b.dtype)
        for i in range(n - 1, -1, -1):
            x[i] = (b[i] - a[i, i + 1:].dot(x[i + 1:])) / a[i, i]
        return x

    @staticmethod
    def LU(a, l, u):
        '''LU decomposition by diag(1) as ones'''

        n = a[0].size
        for i in range(n):
            u[i, i:] = a[i, i:] - l[i, :i].dot(u[:i, i:])
            l[i + 1:, i] = a[i + 1:, i] - l[i + 1:, :i + 1].dot(u[:i + 1, i])
            l[i + 1:, i] /= u[i, i]

    @staticmethod
    def solveLU(l, u, b):
        '''Solves for x when L & U are given
        Depends on LinearSolver.solve'''

        tempx = LinearSolver.solve(l[::-1, ::-1], b[::-1])
        return LinearSolver.solve(u, tempx[::-1])

    def GE(self, pivot=0):
        a = self.a.copy()
        b = self.b.copy()
        n = b.size
        for i in range(0, n - 1):
            if pivot == 1:
                LinearSolver.pivot(a, b, i)

            for j in range(i + 1, n):
                b[j] -= b[i] / a[i, i] * a[j, i]
                a[j] -= a[i] / a[i, i] * a[j, i]

        self.x = LinearSolver.solve(a, b)

    def GE_pivoted(self):
        self.GE(1)

    def doolittle(self):
        self.L, self.U = np.identity(self.b.size), np.identity(self.b.size)

        LinearSolver.LU(self.a, self.L, self.U)
        self.x = LinearSolver.solveLU(self.L, self.U, self.b)

    def crout(self):
        self.L, self.U = np.identity(self.b.size), np.identity(self.b.size)

        LinearSolver.LU(self.a.transpose(), self.L, self.U)
        self.L, self.U = self.U.transpose(), self.L.transpose()
        self.x = LinearSolver.solveLU(self.L, self.U, self.b)

    def cholesky(self):
        n = self.b.size
        self.L = np.identity(n)
        a = self.a
        l, u = self.L, self.L.transpose()
        for i in range(n):
            l[i, i] = sqrt(a[i, i] - np.sum(l[i, :i]**2))
            l[i + 1:, i] = a[i + 1:, i] / l[i, i]
        self.x = LinearSolver.solveLU(l, u, self.b)


def main():
    method_names = ['GE; without pivoting', 'GE; with patial pivoting',
                    'LU decomposition by Doolittle method',
                    'LU decomposition by Crout method',
                    'Cholesky decomposition (for symmetric positive definite matrix)']

    with open("q1.txt", "r") as f:
        n = int(f.readline())
        a, b = np.ndarray(shape=(n, n)), np.ndarray(shape=(n))
        for i, line in enumerate(f):
            a[i] = map(float, line.split()[:-1])
            b[i] = float(line.split()[-1])

    for i, name in enumerate(method_names):
        print 'press ', i, 'for ', name
    choice = int(raw_input())

    solver = LinearSolver(a, b)
    getattr(solver, LinearSolver.methods[choice])()

    with open("out.txt", "w") as fout:
        if solver.x is not None:
            np.savetxt(fout, solver.x, fmt='%f',
                       header='x', footer='\n', comments='')

        if solver.L is not None:
            np.savetxt(fout, solver.L, fmt='%f',
                       header='L', footer='\n', comments='')

        if solver.U is not None:
            np.savetxt(fout, solver.U, fmt='%f',
                       header='U', footer='\n', comments='')


if __name__ == '__main__':
    main()
