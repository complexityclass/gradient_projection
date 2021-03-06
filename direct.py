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

		u =   np.zeros(self.nx)   # u_k+1
		u_1 = np.zeros(self.nx)   # u_k
		u_2 = np.zeros(self.nx)   # u_k-1

		u_2[:] = 0
		u_1[:] = 0

		u[0] = f(self.t[0])
		u_1[0] = f(self.t[1])
		u_2[0] = f(self.t[2])

		res.append(u_2)
		res.append(u_1)

		for n in range(2, self.nt):
			u[0] = f(self.t[n])
			for i in range(1, self.nx - 1):
				u[i] = -u_2[i] + self.gamma * u_1[i - 1] + (2 - 2 * self.gamma) * u_1[i] + self.gamma * u_1[i + 1]

			u[self.nx - 1] = u[self.nx - 2] / (self.dx + 1)

			res.append(list(u))
			u_2 = list(u_1)
			u_1 = list(u)
			#u_2[:], u_1[:] = u_1, u
			u[:] = 0

		return res


	def conjugate(self, v):
		res = [[]]

		p =   np.zeros(self.nx)   # p_k+1
		p_1 = np.zeros(self.nx)   # p_k
		p_2 = np.zeros(self.nx)   # p_k-1

		p_2 = list(v)
		p_2 = [elem * -1 for elem in p_2]
		p_1 = list(p_2)

		p_1[0] = 0
		p_2[0] = 0

		res.append(p_2)
		res.append(p_1)

		for n in range(self.nt - 2, -1, -1):
			p[0] = 0
			for i in range(1, self.nx - 1):
				p[i] = -p_2[i] + self.gamma * p_1[i - 1] + (2 - 2 * self.gamma) * p_1[i] + self.gamma * p_1[i + 1]
			
			p[self.nx - 1] = p[self.nx - 2] / (1 - self.dx)

			res.append(list(p))

			p_2 = list(p_1)
			p_1 = list(p)
			p[:] = 0

		res = reversed(res)

		return list(res)


	def discretezation(self, f, nt, nx):
		res = np.zeros([nt, nx])
		for t in range(nt):
			for k in range(nx):
				res[t,k] = f(t, k)
		return res

	def func_to_arr(self, f, nx):
		res = np.zeros(nx)
		for x in range(nx):
			res[x] = f(x)
		return res



	def gradient_projection(self,u):
		a = self.discretezation(lambda x,y : 1, self.nt, self.nx)
		b = self.discretezation(lambda x,y : 40, self.nt, self.nx)

		u_T = u[self.nt]
		f_x = lambda x : x *sin(x)
		xt = lambda t,x : t * sin(x)
		u_prev = np.zeros([self.nt, self.nx])
		u_next = np.zeros([self.nt, self.nx])

		u_prev = self.discretezation(xt, self.nt, self.nx)

		alpha = 1.
		step = 0

		while True:
			v = list(u_T)
			for index in range(0, self.nx):
				v[index] = v[index] - f_x(self.x[index])

			dj = self.conjugate(v)

			for t in range(self.nt):
				for i in range(self.nx):
					u_next[t, i] = u_prev[t, i] - alpha * 2 * dj[t][i]
					if u_next[t,i] < a[t,i]:
						u_next[t,i] = a[t,i]
					if u_next[t,i] > b[t,i]:
						u_next[t,i] = b[t,i]


			div1 = u_next[self.nt - 1] - u_next[self.nt - 2]
			div1 = [elem / self.dx for elem in div1]

			div2 = u_prev[self.nt - 1] - u_prev[self.nt - 2]
			div2 = [elem / self.dx for elem in div2]

			min_fun1 = div1 - self.func_to_arr(f_x, self.nx)
			min_fun2 = div2 - self.func_to_arr(f_x, self.nx)
			min_fun1 = [elem * elem for elem in min_fun1]
			min_fun2 = [elem * elem for elem in min_fun2]

			if simpson(min_fun1, self.nx, self.dx) > simpson(min_fun2, self.nx, self.dx) :
				u_next = copy(u_prev, self.nt, self.nx)
				alpha *= 0.95

			u_prev = copy(u_next, self.nt, self.nx)
			step += 1

			div_res = u_next[self.nt - 1] - u_next[self.nt - 2]
			min_res = div1 - self.func_to_arr(f_x, self.nx)
			min_res = [elem * elem for elem in min_res]
			print "J: " + simpson(min_res, self.nx, self.dx).__str__() + " alpha = " + alpha.__str__()

			if step == 10000:
				break



def simpson(f, n, h):
	S = f[0] + f[n - 1]

	for i in range(1,n,2):
		S += 4 * f[i]
	for j in range(2,n - 1, 2):
		S += 2 * f[j]

	return S * h / 3


def printArr(arr):
	print "Arr :"
	for row in arr:
		str = ""
		for element in row:
			str = str + " " + element.__str__()
		print str
	print "_______________________________________________________________________"


def copy(arr, nt, nx):
	res = np.zeros([nt, nx])
	for t in range(nt):
		for i in range(nx):
			res[t, i] = arr[t, i]
	return res


if __name__ == '__main__':

	l = 10
	T = 10
	nx = 7
	nt = 7
	
	def f(t):
		return sin(t)

	wave = WaveEq(l, T, nx, nt)
	res = wave.direct(f)
	printArr(res)

	v = res[wave.nt - 1]
	res2 = wave.conjugate(v)


	wave.gradient_projection(res)

	fig = plt.figure()
	plts = []
	plt.hold("off")
	for t in range(wave.nt):
		p, = plt.plot(res[t][:], 'k')
		plts.append([p])
	ani = animation.ArtistAnimation(fig, plts, interval = 90, repeat_delay = 3000)
	plt.show()




 


