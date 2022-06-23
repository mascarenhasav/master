from numpy.lib.npyio import savetxt
import operator
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import os

from pyswarm import pso

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

plt.rcParams['figure.figsize'] = (16, 6)

os.system("rm pops/*")

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
	#print(part.speed)
	part[:] = list(map(operator.add, part, part.speed))

def readLine(path):
	with open(path, 'r') as f:
		return ( float(f.readlines()[-1].split()[0]) )

def evaluate(x):
	os.system("./moon_sop pops")
	fobj = readLine("pops/pop_objs_eval.txt")
	fcons = readLine("pops/pop_cons_eval.txt")

	y = fobj
	#y = fobj + c1(fcons) + c2(x[0]) + c3(x[1])
	if(y > 0):
		y = 0

	#print(f"pos[{x[0], x[1]}]\t fobj:{fobj} fcons:{fcons}")
	return y,

#***************************************
#constraints functions - not working
def c1(x):
	if ( (x > 0) and (x < 0.05) ):
		return
	else:
		return 0.1
def c2(x):
	if( (x > 0) and (x < 1) ):
		return 0
	else:
		return 0.1
def c3(x):
	if(  (x > 0) and (x < 1) ):
		return 0
	else:
		return 0.1

#***************************************
# parameters
size = 2

pmin = 0
pmax = 1
smin = 0
smax = 0.5

phi1 = 1
phi2 = 1
omega = 0.5

npop = int(input("Numero de individuos: "))
GEN = int(input("Numero de geracoes: "))
N = 1  # number of tentatives


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

	best = None
	bestG = None
	fit = 0
	gen = [i for i in range(GEN)]

	posx = [[] for i in range(npop)]
	posy = [[] for i in range(npop)]

	for j in range(N):
		pop = toolbox.population(n=npop)
		error = []
		line = f'{random.uniform(0, 1)}\t{random.uniform(0, 1)}\n'
		with open("./pops/pop_vars_eval.txt", 'a+') as f:
			f.write(line)

		for g in range(GEN):
			for i in range(npop):
				pop[i].fitness.values = toolbox.evaluate(pop[i])
				#print(pop[i].fitness.values)

				if(  (pop[i][0] < 0) or (pop[i][0] > 1) ):
					pop[i][0] = 0.5
				if(  (pop[i][1] < 0) or (pop[i][1] > 1) ):
					pop[i][1] = 0.5

				if not pop[i].best or pop[i].best.fitness < pop[i].fitness:
					pop[i].best = creator.Particle(pop[i])
					pop[i].best.fitness.values = pop[i].fitness.values
				if not best or best.fitness < pop[i].fitness:
					best = creator.Particle(pop[i])
					best.fitness.values = pop[i].fitness.values

				part = pop[i]

				'''
				print(f'part: {part}\t{part.fitness.values}')
				print(f'pbest:{part.best}\t{part.best.fitness.values}')
				print(f'best: {best}\t{best.fitness.values}')
				print(f'---------------------------------------------')
				'''

				line = f'{part[0]}\t{part[1]}\n'
				with open("./pops/pop_vars_eval.txt", 'a+') as f:
					f.write(line)

				print(part.fitness)
				error.append(part.fitness.values[0])
				if(part.fitness.values[0] <= fit):
					fit = part.fitness.values[0]
					bestGen = part

			for part in pop:
				toolbox.update(part, best)


			for i in range(npop):
				posx[i].append(pop[i][0])
				posy[i].append(pop[i][1])
				#print(f"{posx[i][g]} {posy[i][g]}")

			std = np.std(error)
			min = np.min(error)
			max = np.max(error)
			mean = np.mean(error)
			error = []
			print(f"Gen:{g} bestGen:{bestGen} mean:{mean:.3f} min:{min:.2f} max:{max:.2f} std:{std:.2f}")
			print(f"Gen:{g} best:   {best}")
			#errorN.append(best.fitness.values[0])

			# Gather all the fitnesses in one list and print the stats
			logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
			#print(logbook.stream, best)
			#print(best, best.fitness.values[0])


		#print(posx)
		#print(posy)


		print(best, best.fitness.values)

		'''
		maxError = max(error)
		minError = min(error)

		if j==0:
			errorG = maxError

		plt.plot(gen, error, label=f'n={j}, min:{min(error)}')

		if minError < errorG:
			errorG = minError
			bestG = best


		del part.fitness.values
		del best.fitness.values
		'''
		#print(f"best: {bestG}  value: {errorG}")

	for g in range(npop):
		plt.scatter(posx[g], posy[g])
	plt.xlim(0, 1)
	plt.ylim(0, 1)
	plt.title(f"PSO with {npop} individuals and {GEN} generations\npos:{best} min:{best.fitness.values[0]}", fontsize=22)
	plt.grid(True)
	plt.ylabel("Latitude", fontsize=22)
	plt.xlabel("Longitude", fontsize=22)
	plt.savefig("errors.png")

	return pop, logbook, best


if __name__ == "__main__":
	main()
