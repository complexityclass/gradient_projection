import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.animation as animation
from math import sin

class WaveEq(object):

	def __init__(self, l, T, nx, nt):

		self.x = np.linspace(0, l, nx)
		self.t = np.linspace(0, T, nt)

		self.dx = self.x[1] - self.x[0]
		self.dt = self.t[1] - self.t[0]
		self.nx = nx
		self.nt = nt
		self.gamma = (self.dx ** 2) / (self.dt ** 2)

	def solve(self, f):

		res = [[]]

		u =   np.zeros(self.nx + 1)   # u_k+1
		u_1 = np.zeros(self.nx + 1)   # u_k
		u_2 = np.zeros(self.nx + 1)   # u_k-1

		u_2[:] = 0
		u_1[:] = 0

		u[0] = f(self.t[0])
		u_1[0] = f(self.t[1])
		u_2[0] = f(self.t[2])

		for n in range(1, self.nt):
			for i in range(1, self.nx):
				u[i] = -u_2[i] + self.gamma * u_1[i - 1] + (2 - 2 * self.gamma) * u_1[i] + self.gamma * u_1[i + 1]

			u[self.nx] = u[self.nx - 1] / (self.dx + 1)
			u[0] = f(self.t[i])

			res.append(u)

			u_2[:], u_1[:] = u_1, u

		return res, u, self.x, self.t


if __name__ == '__main__':

	l = 5
	T = 5
	nx = 5
	nt = 5
	
	def f(t):
		return sin(t)

	wave = WaveEq(l, T, nx, nt)

	res, u, x, t = wave.solve(f)

	print res


