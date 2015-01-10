import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.animation as animation
from math import sin
from math import exp
from mpl_toolkits.mplot3d import Axes3D

class WaveEq(object):

	def __init__(self, l, T, nx, nt):

		self.x = np.linspace(0, l, num = nx)
		self.t = np.linspace(0, T, num = nt)

		self.dx = self.x[1] - self.x[0]
		self.dt = self.t[1] - self.t[0]
		self.nx = nx
		self.nt = nt
		self.gamma = (self.dx ** 2) / (self.dt ** 2)
		

	def direct(self, f):

		res = [[]]

		u =   np.zeros(self.nx + 1)   # u_k+1
		u_1 = np.zeros(self.nx + 1)   # u_k
		u_2 = np.zeros(self.nx + 1)   # u_k-1

		u_2[:] = 0
		u_1[:] = 0

		u[0] = f(self.t[0])
		u_1[0] = f(self.t[1])
		u_2[0] = f(self.t[2])

		res.append(u_1)
		res.append(u_2)

		for n in range(2, self.nt):
			for i in range(1, self.nx):
				u[i] = -u_2[i] + self.gamma * u_1[i - 1] + (2 - 2 * self.gamma) * u_1[i] + self.gamma * u_1[i + 1]

			u[self.nx - 1] = u[self.nx - 2] / (self.dx + 1)
			u[0] = f(self.t[n])

			res.append(u)

			u_2[:], u_1[:] = u_1, u


		return res


	def conjugate(self, v):
		res = [[]]

		p =   np.zeros(self.nx + 1)   # p_k+1
		p_1 = np.zeros(self.nx + 1)   # p_k
		p_2 = np.zeros(self.nx + 1)   # p_k-1

		p[0] = p_1[0] = p_2[0] = 0
		p_2[:] = v
		p_1[:] = v

		res.append(p_2)
		res.append(p_1)

		for n in range(self.nt - 2, -1, -1):
			for i in range(1, self.nx - 1):
				p[i] = -p_2[i] + self.gamma * p_1[i - 1] + (2 - 2 * self.gamma) * p_1[i] + self.gamma * p_1[i + 1]
			p[self.nx - 1] = -p[self.nx - 2] / (self.dx)

			res.append(p)
			p_2[:], p_1[:] = p_1, p

		res = reversed(res)

		return res


	def gradient_projection(self,u):
		a = lambda x : 20
		b = lambda x : 40

		f_x = u[self.nt]
		u_prev = lambda t : sin(t)
		u_next = np.zeros([self.nt, self.nx])
		lmbd = 1.
		st = 0

		index = 0

		while index < 3:
			direct_system = self.direct(u_prev)
			v = direct_system[self.nt]
			for idx in range(1, self.nx):
				v[idx] = v[idx] - f_x[idx]

			conjugate_system = self.conjugate(v)

			printArr(conjugate_system)

			index = index + 1



def printArr(arr):
	for row in arr:
		str = ""
		for element in row:
			str = str + " " + element.__str__()
		print str
		print "_______________________________________________________________________"




if __name__ == '__main__':

	l = 5
	T = 5
	nx = 7
	nt = 7
	
	def f(t):
		return sin(t)

	wave = WaveEq(l, T, nx, nt)
	#res = wave.direct(f)
	#printArr(res)

	#u = res[wave.nt - 1]
	#res2 = wave.conjugate(v)

	#printArr(res2)

	#wave.gradient_projection(res)

	#fig = plt.figure()
	#plts = []
	#plt.hold("off")
	#for t in range(wave.nt):
	#	p, = plt.plot(res[t][:], 'k')
	#	plts.append([p])
	#ani = animation.ArtistAnimation(fig, plts, interval = 500, repeat_delay = 3000)
	#plt.show()




 


