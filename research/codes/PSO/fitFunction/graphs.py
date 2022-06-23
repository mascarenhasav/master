import matplotlib.pyplot as plt
import numpy as np

#1-polynomial   2-fourier
tipo = int(input("1- Polynomial app     2- Fourier app \n"))

t = np.linspace(0, 6*np.pi, 500)
f = np.cos(t)
f += np.random.normal(0, 0.2, 500)

data = open("bestG2.txt").read().replace('\n', ' ').split(' ')[:-1]
bestG2 = [float(value) for value in data]
data = open("bestG.txt").read().replace('\n', ' ').split(' ')[:-1]
bestG = [float(value) for value in data]
print("***************************************************")
if(tipo == 1):
	print("Cossine approximation by a sixth degree polynomial")
	print("of the type c6x⁶ + c5x⁵ + c4x⁴ + c3x³ + c2x² + c1x¹ + c0")
	print("\nCoefficients found by PSO algorithm was")
	print(f"c6:{bestG2[6]}\nc5:{bestG2[5]}\nc4:{bestG2[4]}\nc3:{bestG2[3]}\nc2:{bestG2[2]}\nc1:{bestG2[1]}\nc0:{bestG2[0]}\n")
	plt.title("Cos by Sixth degree Polynomial approximation", fontsize = 15)
elif(tipo == 2):
	print("Cossine approximation by a sixth degree of Fourier's series")
	print("of the type c6cos(3pix/L) + c5sin(3pix/L) + c4cos(2pix/L) + c3sin(2pix/L) + c2cos(pix/L) + c1sin(pix/L) + c0/2")
	print("\nCoefficients found by PSO algorithm was")
	print(f"c6:{bestG2[6]}\nc5:{bestG2[5]}\nc4:{bestG2[4]}\nc3:{bestG2[3]}\nc2:{bestG2[2]}\nc1:{bestG2[1]}\nc0:{bestG2[0]}\n")
	plt.title("Cos by Sixth degree Fourier's series approximation", fontsize = 15)

plt.xlabel("t", fontsize = 15)
plt.ylabel("cos(t)", fontsize = 15)
plt.plot(t, np.cos(t), c='g', label='cos', linewidth=3)
plt.scatter(t, f, label="data")
plt.plot(t, eq(bestG, t, tipo), c='r', label="Approx - DEAP", linewidth=2)
plt.plot(t, eq(bestG2, t, tipo), c='y', label="Approx - Pyswarm", linewidth=2)
plt.grid(True)
plt.legend()
plt.show()
