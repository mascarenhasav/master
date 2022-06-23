from numpy.lib.npyio import savetxt
import operator
import random
import numpy as np
import math
import matplotlib.pyplot as plt

from pyswarm import pso

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

plt.rcParams['figure.figsize'] = (16, 6)


#****************************************************
# Functions for PSO with DEAP
def generate(size, pmin, pmax, smin, smax):
	part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
	part.speed = [random.uniform(smin, smax) for _ in range(size)]
	part.smin = smin
	part.smax = smax
	return part

def updateParticle(part, best, phi1, phi2, omega):
	w  = (random.uniform(0, omega) for _ in range(len(part)))
	#c1 = (phi1 for _ in range(len(part)))
	#c2 = (phi2 for _ in range(len(part)))
	c1 = (random.uniform(0, phi1) for _ in range(len(part)))
	c2 = (random.uniform(0, phi2) for _ in range(len(part)))
	vPart  = map(operator.mul, c1, map(operator.sub, part.best, part))
	vSwarm = map(operator.mul, c2, map(operator.sub, best, part))
	part.speed = list(map(operator.add, map(operator.mul, w, part.speed), map(operator.add, vPart, vSwarm)))
	for i, speed in enumerate(part.speed):
		if abs(speed) < part.smin:
			#part.speed[i] = math.copysign(part.smin, speed)
			part.speed[i] = part.smin
		elif abs(speed) > part.smax:
			#part.speed[i] = math.copysign(part.smax, speed)
			part.speed[i] = part.smax
	part[:] = list(map(operator.add, part, part.speed))

def evaluate(x):
	return fobj(x),

#**************************************
#objective functions

#size = 6
def fobj(x):
	model = eq(x, t, tipo)
	return rmse(model, t) + g1(x[5]) + g2(x[4])
def eq(x, t, func):
	if(func == 1):
		return pol(x, t)
	elif(func == 2):
		return fourier(x, t)

#polynom which will be approximated
def pol(x, t):
	return x[6]*(t**6) + x[5]*(t**5) + x[4]*(t**4) + x[3]*(t**3) + x[2]*(t**2) + x[1]*(t**1) + x[0]*(t**0)
#fourier
def fourier(x, t):
	#L = len(t)/2
	N = size-1
	L = 3*np.pi
	result = 0
	i = 1
	n = 1
	for _ in range(1, int(N/2)+1):
		result += x[n+1]*np.cos(i*np.pi*t/L) + x[n]*np.sin(i*np.pi*t/L)
		i += 1
		n += 2
	return result

#return the mean square error
def rmse(model, t):
	return np.sqrt( sum( ((model - f)**2)/len(t) ) )

#***************************************
#constraints functions - not working
def g1(x):
	if( (x >= -0.5) and (x <= 0.5) ):
		return 0
	else:
		return 10000

def g2(x):
	if( (x >= -0.5) and (x <= 0.5) ):
		return 0
	else:
		return 10000


#***************************************
# parameters
size = 7

pmin = -1
pmax = 1
smin = -1
smax = 1

phi1 = 1
phi2 = 1
omega = 0.7

npop = 42
GEN = 30
N = 10  # number of tentatives

lb = [smin for i in range(size)]
ub = [smax for i in range(size)]
error_Pyswarm = []



#***************************************
#functino
t = np.linspace(0, 6*np.pi, 500)
f = np.cos(t)
f += np.random.normal(0, 0.2, 500)

#1-polynomial   2-fourier
tipo = int(input("1- Polynomial app     2- Fourier app \n"))

#***************************************
# initialization of the operators in deap
creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) #-1 means minimize
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, 
    smin=None, smax=None, best=None)
toolbox = base.Toolbox()
toolbox.register("particle", generate, size=size, pmin=pmin, pmax=pmax, smin=smin, smax=smax)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=phi1, phi2=phi2, omega=omega)
toolbox.register("evaluate",  evaluate)

def main():
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", np.mean)
	stats.register("std", np.std)
	stats.register("min", np.min)
	stats.register("max", np.max)

	logbook = tools.Logbook()
	logbook.header = ["gen", "evals"] + stats.fields

	errorG_Pyswarm = 10000
	best = None
	bestG = None
	errorG = None
	gen = [i for i in range(GEN)]


	for j in range(N):
		pop = toolbox.population(n=npop)
		error = []

		for g in range(GEN):
			for part in pop:
				part.fitness.values = toolbox.evaluate(part)
				#print(part.fitness.values)
				if not part.best or part.best.fitness < part.fitness:
					part.best = creator.Particle(part)
					part.best.fitness.values = part.fitness.values
				if not best or best.fitness < part.fitness:
					best = creator.Particle(part)
					best.fitness.values = part.fitness.values
			for part in pop:
				toolbox.update(part, best)

			error.append(best.fitness.values[0])

			# Gather all the fitnesses in one list and print the stats
			logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
			#print(logbook.stream, best)

		xopt, fopt = pso(fobj, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, swarmsize=npop, 
		         omega=omega, phip=phi1, phig=phi2, maxiter=GEN, debug=False)
		error_Pyswarm.append(fopt)
		#print(xopt, fopt)
		if(error_Pyswarm[-1] < errorG_Pyswarm):
			errorG_Pyswarm = error_Pyswarm[-1]
			xoptG = xopt


		maxError = max(error)
		minError = min(error)

		if j==0:
			errorG = maxError

		#error = [i/1000000 for i in error]
		plt.plot(gen, error, label=f'n={j}, min:{min(error)}')

		if minError < errorG:
			errorG = minError
			bestG = best

		del part.fitness.values
		del best.fitness.values
		print(f"best: {bestG}  value: {errorG}")


	plt.ylabel(f"Erro", fontsize=20)
	plt.xlabel("Generations", fontsize=15)
	plt.xlim(0, GEN)
	plt.title(f"Erro x Generations   population={npop}", fontsize=15)
	plt.grid(True)
	plt.legend()
	plt.show()

	savetxt("bestG.txt", bestG)
	savetxt("bestG2.txt", xopt)
	print(f'\nMenor erro encontrado DEAP: {errorG}')
	print(f'Menor erro encontrado Pyswarm: {errorG_Pyswarm}')

	return pop, logbook, best


if __name__ == "__main__":
	main()
