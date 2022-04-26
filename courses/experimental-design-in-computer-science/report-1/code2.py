'''
Importing libraries
'''
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date
import numpy as np
import sys



print ("*********************************************************")
print ("Mascarenhas Alexandre")
print ("Experimental Design in Computer Science 2022/1\nUniversity of Tsukuba")
print ("Report 1 - code 2")
print ("*********************************************************")

plt.rcParams['figure.figsize'] = (14, 8)

'''
importing datasets
'''

naruto = pd.read_csv("faria-lima.csv", delimiter = ',', skiprows=1) #import dataset with number of cyclists
luffy = pd.read_csv("temperature-sp.csv", delimiter = '\t', skiprows=18) #import dataset with temperature



'''
Variables to configure the day of the week and number of datas
'''

nInit = 713
nEnd = 1444
nMed = 6

print ("Enter the number of day of the week:\n")
print ("   * 0 -> Mon")
print ("   * 1 -> Tue")
print ("   * 2 -> Wed")
print ("   * 3 -> Thu")
print ("   * 4 -> Fri")
print ("   * 5 -> Sat")
print ("   * 6 -> Sun\n")

dayWeek = int(input("Day:"))

dateBikers = naruto.iloc[nInit+dayWeek:nEnd, 0]
bikersIbira = naruto.iloc[nInit+dayWeek:nEnd, 2]
bikersPinha = naruto.iloc[nInit+dayWeek:nEnd, 3]

dateTemp = luffy.iloc[nInit+dayWeek:nEnd, 0]
tempMax = luffy.iloc[nInit+dayWeek:nEnd, 4]
tempMed = luffy.iloc[nInit+dayWeek:nEnd, 5]

dayWeekStr = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

print (dateTemp)



'''

'''

n = 0
newTempMed = []
newTempMax = []
newDateTemp = []
newBikersIbira = []
newBikersPinha = []
m = nInit+dayWeek

for i in tempMed:
	if(n == nMed):
		newTempMed.append(i)
		newTempMax.append(tempMax[m])
		newDateTemp.append(dateTemp[m])
		newBikersIbira.append(bikersIbira[m])
		newBikersPinha.append(bikersPinha[m])
		n = 0
	else:
		n += 1

	m += 1



'''
Setting up the graph
'''

fig, ax1 = plt.subplots()
plt.grid(True)

ax1.set_xlabel("Date")
ax1.set_ylabel("Number of Cyclists (N)")
ax1.set_title('Daily Temperature and Daily Number of Cyclists every '+dayWeekStr[dayWeek]+' between Jan/2018 and Jan/2020')
ax1.tick_params(axis='x', labelrotation=90)
plot1 = ax1.plot(newDateTemp, newBikersIbira, 'g', label="Number of Cyclists (N)")
plot2 = ax1.plot(newDateTemp, newBikersPinha, 'b', label="Number of Cyclists (N)")

ax2 = ax1.twinx()
ax2.set_ylabel("Temperature (C)")
plot3 = ax2.plot(newDateTemp, newTempMed, 'r', label="Average Temperature (C)")

lns = plot1 + plot2 + plot3
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, loc=0)

plt.savefig(f'./CycliXTemp-2018_20-'+dayWeekStr[dayWeek]+'.png', format='png')



'''
Finishing the script
'''

sys.exit()
