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
nEnd = 300
nMed = 6

dateBikers = naruto.iloc[nInit:nEnd, 0]
bikersIbira = naruto.iloc[nInit:nEnd, 2]
bikersPinha = naruto.iloc[nInit:nEnd, 3]

dateTemp = luffy.iloc[nInit:nEnd, 0]
tempMax = luffy.iloc[nInit:nEnd, 4]
tempMed = luffy.iloc[nInit:nEnd, 5]

print (dateTemp)
print (dateBikers)

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
newBikersIbira = []
m = 0


for i in tempMed:
	if(n == nMed):
		newTempMed.append(i)
		newTempMax.append(tempMax[m])
		newDateTemp.append(dateTemp[m])
		newBikersIbira.append(bikersIbira[m])
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
plot1 = ax1.bar(newTempMed, newBikersIbira)

plt.show()
#plt.savefig('./testeVIB.png', format='png')
