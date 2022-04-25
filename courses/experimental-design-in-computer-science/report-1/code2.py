import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date
import numpy as np
import scipy as sp
from sklearn.linear_model import LinearRegression
from scipy import optimize
#matplotlib.use('Agg')

now = datetime.now()
now = now.strftime("%d-%m-20%y")

print ("Report 1")
print ("Mascarenhas Alexandre\nUniversity of Tsukuba")

plt.rcParams['figure.figsize'] = (14, 6)

naruto = pd.read_csv("faria-lima.csv", delimiter = ',', skiprows=1)
luffy = pd.read_csv("temperature-sp.csv", delimiter = '\t', skiprows=18)


nInit = 5
nEnd = 150
nMed = 6

dateBikers = naruto.iloc[nInit:nEnd, 0]
bikersIbira = naruto.iloc[nInit:nEnd, 2]
bikersPinha = naruto.iloc[nInit:nEnd, 3]

dateTemp = luffy.iloc[nInit:nEnd, 0]
tempMax = luffy.iloc[nInit:nEnd, 4]
chuva = luffy.iloc[nInit:nEnd, 3]
tempMed = luffy.iloc[nInit:nEnd, 5]

print (dateTemp)
print (dateBikers)

print (luffy)

#tempMed = tempMed.mul(50);
'''
print (date)
print (bikersIbira)
print (bikersPinha)
print (dateTemp)
print (tempMed)
'''
n = 0
newTempMed = []
newTempMax = []
newDateTemp = []
newRain = []
newBikersIbira = []
m = 0


for i in tempMed:
	if(n == nMed):
		newTempMed.append(i)
		newTempMax.append(tempMax[m])
		newDateTemp.append(dateTemp[m])
		newBikersIbira.append(bikersIbira[m])
		newRain.append(chuva[m])
		n = 0
	else:
		n += 1

	m += 1



#print (newTempMed)
#print (newTempMax)
#print (newDateTemp)
#print (newBikersIbira)

fig, ax1 = plt.subplots()
plt.grid(True)
ax1.set_xlabel("Date")
ax1.set_ylabel("Num Cyclists")
ax1.set_title('Number of Cyclists x Date')
ax1.tick_params(axis='x', labelrotation=45)
plot1 = ax1.plot(newDateTemp, newBikersIbira, 'g', label="Num Cyclists")

ax2 = ax1.twinx()
ax2.set_ylabel("Temperature (C)")
plot2 = ax2.plot(newDateTemp, newTempMed, 'r', label="TempMed (C)")

'''
ax3 = ax1.twinx()
ax3.set_ylabel("Temperature (C)")
ax3.plot(newDateTemp, newRain, 'b', label="TempMed (C)")
'''

lns = plot1 + plot2
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, loc=0)


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

