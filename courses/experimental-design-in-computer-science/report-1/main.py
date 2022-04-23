import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date
import numpy as np
from scipy import fftpack as FFT

#matplotlib.use('Agg')

now = datetime.now()
now = now.strftime("%d-%m-20%y")

print ("Report 1")
print ("Mascarenhas Alexandre\nUniversity of Tsukuba")

plt.rcParams['figure.figsize'] = (14,10)

naruto = pd.read_csv("faria-lima.csv", delimiter = ',', skiprows=[0])
luffy = pd.read_csv("temperature-sp.csv", delimiter = '\t', skiprows=[0])

nDados = 800
nMed = 30

date = naruto.iloc[:nDados, 0]
bikersIbira = naruto.iloc[:nDados, 2]
bikersPinha = naruto.iloc[:nDados, 3]

dateTemp = luffy.iloc[:nDados, 0]
tempMed = luffy.iloc[:nDados, 5]
#tempMed = tempMed.mul(50);
'''
print (date)
print (bikersIbira)
print (bikersPinha)
print (dateTemp)
print (tempMed)
'''
n = 0
sumTemp = 0
tempMedMed = []
dateTempMed = []
bikersMed = []
sumBikers = 0
m = 0
for i in tempMed:
	if(n == nMed):
		sumTemp /= nMed
		sumBikers /= nMed
		tempMedMed.append(sumTemp)
		bikersMed.append(sumBikers)
		dateTempMed.append(dateTemp[m])
		sumTemp = 0
		sumBikers = 0
		n = 0
	else:
		sumTemp += i
		sumBikers += bikersIbira[m]
		n += 1
	m += 1



print (tempMedMed)
print (bikersMed)
print (dateTempMed)

fig, ax1 = plt.subplots()
plt.grid(True)
ax1.set_xlabel("Date")
ax1.set_ylabel("N Bikers")
ax1.set_title('Number of Bikers')
plot1 = ax1.plot(dateTempMed, bikersMed, 'r', label="N Bikers")

ax2 = ax1.twinx()
ax2.set_ylabel("Temperature (C)")
plot2 = ax2.plot(dateTempMed, tempMedMed, label="Temp (C)")

lns = plot1 + plot2
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, loc=0)

ax1.tick_params(axis='x', labelrotation=45)

#plt.plot(date, bikersPinha, 'r')
plt.show()
#plt.savefig('./testeVIB.png', format='png')


#print(naruto)


'''
maxTensao = eixoY2.max()
minTensao = eixoY2.min()
eixoY2 = list(map(lambda x:((x-offsetTensao2)*46.06), eixoY2))

#eixoY1tempo = eixoY1
eixoY2tempo = eixoY2

#eixoY1 = np.asarray(eixoY1)
#eixoY2 = np.asarray(eixoY2)

#rangeAmostra = eixoY1.size

#rangeFreq = np.linspace(0.0, sps/div, int(rangeAmostra/div))

#freqY1 = FFT.fft(eixoY1)
#freqY2 = FFT.fft(eixoY2)

#yY1 = (4/rangeAmostra) * np.abs(freqY1[0:np.int(rangeAmostra/div)])
#yY2 = (4/rangeAmostra) * np.abs(freqY2[0:np.int(rangeAmostra/div)])



#tempo = naruto.iloc[1:3500, 0]
#print (corrente)
#print (tempo)



plt.grid(True)
plt.title('Curva de vibracao - teste sensorIEPE')
plt.xlabel('Tempo (ms)')
plt.ylabel('Vibracao (G)')
plt.plot(eixoX, eixoY)
plt.plot(eixoX, eixoY2, 'r')
plt.axis([7500, 8200, -8, 8])
plt.savefig('./testeVIB.png', format='png')


fig1=plt.figure()
plt.subplot(4,1,1)
plt.grid(True)
plt.title('Curva de vibracao - teste sensorIEPE')
plt.xlabel('Frequencia (Hz)')
plt.ylabel('Vibracao (G)')
plt.plot(rangeFreq, yY1, 'r')
plt.axis([1, sps/div, 0, 4])

plt.subplot(4,1,2)
plt.grid(True)
plt.title('Curva de vibracao - teste sensorIEPE')
plt.xlabel('Tempo (ms)')
plt.ylabel('Vibracao (G)')
plt.plot(eixoX, eixoY1tempo, 'r')
#plt.axis([7200, 7800, -8, 8])
plt.axis([0, 10000, -8, 8])

plt.subplot(4,1,3)
plt.grid(True)
plt.title('Curva de vibracao - teste sensorIEPE')
plt.xlabel('Frequencia (Hz)')
plt.ylabel('Vibracao (G)')
plt.plot(rangeFreq, yY2)
plt.axis([1, sps/div, 0, 4])




plt.subplot(2,1,1)
plt.grid(True)
plt.title('Curva de vibracao - teste sensorIEPE MDL')
plt.xlabel('Tempo (ms)')
plt.ylabel('Vibracao (G)')
plt.plot(eixoX, eixoY2tempo, 'r')
#plt.axis([7200, 7800, -8, 8])
plt.axis([3000, 5000, -75, 75])
#plt.savefig('./testeVIB-MDL.png', format='png')



plt.subplot(2,1,2)
plt.grid(True)
plt.title('Curva de vibracao - teste sensorIEPE SKF')
plt.xlabel('Tempo (ms)')
plt.ylabel('Vibracao (G)')
plt.plot(eixoXSKF, eixoYteste)
#plt.axis([7200, 7800, -8, 8])
plt.axis([2, 3.2, -75, 75])
plt.savefig('./testeVIB-SKF.png', format='png')
'''
#plt.show()

