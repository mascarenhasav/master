'''
Importing libraries
'''
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date
import numpy as np
import sys
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# Date time conversion registration
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

print ("*********************************************************")
print ("Mascarenhas Alexandre")
print ("Experimental Design in Computer Science 2022/1\nUniversity of Tsukuba")
print ("Report 1 - code 1")
print ("*********************************************************")

plt.rcParams['figure.figsize'] = (14, 8)

'''
importing datasets
'''

naruto = pd.read_csv("faria-lima.csv", delimiter = ',', skiprows=1) #import dataset with number of cyclists
luffy = pd.read_csv("temperature-sp.csv", delimiter = '\t', skiprows=18) #import dataset with temperature
luffy = pd.read_csv("temperature-sp.csv", delimiter = '\t',usecols=['Data Medicao','TEMPERATURA MEDIA COMPENSADA, DIARIA(°C)'], parse_dates=['Data Medicao']) #import dataset with temperature
luffy.set_index('Data Medicao',inplace=True)

print (luffy)

'''
Variables to configure the day of the week and number of datas
'''
'''
#nInit = 713 2018
nInit = 1077
nEnd = 1444
nMed = 6

dateBikers = naruto.iloc[nInit:nEnd, 0]
bikersIbira = naruto.iloc[nInit:nEnd, 2]
bikersPinha = naruto.iloc[nInit:nEnd, 3]

dateTemp = luffy.iloc[nInit:nEnd, 0]
tempMed = luffy.iloc[nInit:nEnd, 1]

dayWeekStr = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

print (dateTemp)
'''
'''
Setting up the graph
'''

fig, ax = plt.subplots()
#luffy.plot(ax=ax)
data = luffy["2019-01-01":"2020-01-01"]
print (data)
ax.bar(data.index.values, data['TEMPERATURA MEDIA COMPENSADA, DIARIA(°C)'].values)
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.grid(False)
'''
ax1.set_xlabel("Date")
ax1.set_ylabel("Number of Cyclists (N)")
ax1.set_title('Daily Temperature and Daily Number of Cyclists between Jan/2018 and Jan/2020')
ax1.tick_params(axis='x', labelrotation=90)
plot1 = ax1.plot(dateTemp, bikersIbira, 'g', label="Number of Cyclists towards Ibirapuera (N)")
plot2 = ax1.plot(dateTemp, bikersPinha, 'b', label="Number of Cyclists towards Pinheiros (N)")
plot1 = ax1.bar(dateTemp, bikersIbira, 'g')

ax2 = ax1.twinx()
ax2.set_ylabel("Temperature (C)")
plot3 = ax2.plot(dateTemp, tempMed, 'r', label="Average Temperature (C)")
ax2.plot(dateTemp, tempMed, 'r', label="Average Temperature (C)")

date_form = DateFormatter("%m/%d")
#ax1.xaxis.set_major_formatter(date_form)
#ax2.xaxis.set_major_formatter(date_form)
#set ticks every week
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
#set major ticks format
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
'''
ax.xaxis.set_major_locator(mdates.WeekdayLocator())
#set major ticks format
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

'''
lns = plot1 + plot2 + plot3
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, loc=0)
'''
plt.savefig(f'./CycliXTemp-2018_20.png', format='png')



'''
Finishing the script
'''

sys.exit()
