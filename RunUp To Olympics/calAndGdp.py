import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
# import re
from scipy.stats import linregress

def fileSaver(answer, colNames, filename):
    answer = pd.DataFrame(answer)
    answer.columns = colNames
    answer.to_csv(filename, index=False) 

#col 2 has gold medals, 3 has silver medals and 4th has bronze medals
medalData = pd.read_csv('data/medalsTally.csv')
medalArr = medalData.to_numpy()
calData = pd.read_csv('data/calorie-intake.csv')
calArr = calData.to_numpy()

#store overall Medals  from 2008 to 2013 for these countries
medalDict = {}
for i in range(0, len(medalArr)):
    medalArr[i][6] = medalArr[i][6].replace(',','')
    medalDict[medalArr[i][0]] = int(medalArr[i][6])

#store 5 year avg cal consumption from 2008 to 2013 for these countries
calDict = {}
for i in range(52, len(calArr), 53):
    calDict[calArr[i][0]] = 0
    stride = 5
    for j in range(i,i-stride,-1):
        calDict[calArr[i][0]]+=calArr[j][7]
    calDict[calArr[i][0]] = calDict[calArr[i][0]]/stride

x = []
y = []

#some data merging and cleaning; we don't consider medals<10:
#divide medals between aus and new zeland based on medals they already won
total = medalDict['Australia']+medalDict['New Zealand']
medalDict['Australia']+= int(medalDict['Australasia']*medalDict['Australia']/total)
medalDict['New Zealand']+= int(medalDict['Australasia']*medalDict['New Zealand']/total)

#discard barain and bohemia, 
#combine a few historic countries' data into modern ones
medalDict['Czechia'] = medalDict['Czech Republic']+medalDict['Czechoslovakia']+medalDict['Slovakia']
medalDict['Germany'] = medalDict['United Team of Germany']+medalDict['East Germany']+medalDict['West Germany']
medalDict['United Kingdom'] = medalDict['Great Britain']
medalDict['Russia'] = medalDict['Soviet Union']+medalDict['Russian Empire']+medalDict['ROC']
medalDict['Taiwan'] = medalDict['Chinese Taipei']

z = []

for key in medalDict:
    if key not in calDict.keys():
        # print(key, medalDict[key])
        continue
    else:
        x.append(medalDict[key])
        y.append(calDict[key])
        z.append(key)

x, y = np.array(x), np.array(y)
slope, intercept, r_value, p_value, std_err = linregress(x, y)
print("The r value before introduction of population in the picture is:", r_value)

popData = pd.read_csv('data/population.csv', error_bad_lines=False)
popArr = popData.to_numpy()
popDict = {}
for row in popArr:
    popDict[row[0]] = row[-1]

#treating mising data for 13 countries
popDict['Bahamas'] = popDict['Bahamas, The']
popDict['Egypt'] = popDict['Egypt, Arab Rep.']
popDict['Iran'] = popDict['Iran, Islamic Rep.']
popDict['Hong Kong'] = popDict['Hong Kong SAR, China']
popDict['North Korea'] = popDict["Korea, Dem. People's Rep."]
popDict['South Korea'] = popDict['Korea, Rep.']
popDict['Kyrgyzstan'] = popDict['Kyrgyz Republic']
popDict['Russia'] = popDict['Russian Federation']
popDict['Venezuela'] = popDict['Venezuela, RB']
popDict['Czechia'] = popDict['Czech Republic']
popDict['Taiwan'] = 23600000

gdpDict = {}
def isNan(val):
    return val!=val

gdpData = pd.read_csv('data/gdp_per_capita.csv')
gdpArr = gdpData.to_numpy()

for row in gdpArr:
    gdpDict[row[0]] = 0 if isNan(row[64]) else float(row[64]) 

#treating mising data for 13 countries
gdpDict['Bahamas'] = gdpDict['Bahamas, The']
popDict['Egypt'] = gdpDict['Egypt, Arab Rep.']
gdpDict['Iran'] = gdpDict['Iran, Islamic Rep.']
gdpDict['Hong Kong'] = gdpDict['Hong Kong SAR, China']
gdpDict['North Korea'] = gdpDict["Korea, Dem. People's Rep."]
gdpDict['South Korea'] = gdpDict['Korea, Rep.']
gdpDict['Kyrgyzstan'] = gdpDict['Kyrgyz Republic']
gdpDict['Russia'] = gdpDict['Russian Federation']
gdpDict['Venezuela'] = gdpDict['Venezuela, RB']
gdpDict['Czechia'] = gdpDict['Czech Republic']
gdpDict['Taiwan'] = 30000



pop = []
cal = []

y = []
gdp = []
z = []

for key in medalDict:
    if key not in calDict.keys() or key not in popDict.keys() or key not in gdpDict.keys():
        # print(key, medalDict[key])
        continue
    else:
        y.append(int(medalDict[key]))
        cal.append(float(calDict[key]))
        z.append(key)
        pop.append(int(popDict[key]))
        gdp.append(int(gdpDict[key]))

pop = np.array(pop)
cal = np.array(cal)
mini = cal.min()
maxi = cal.max()
cal = (cal-mini)/(maxi-mini)
pop = pop
norm = np.linalg.norm(cal)

max_pow = -1
max_r = 0
pltx = []
plty = []
for i in range(0,7):
    x = pop*pow(cal, i)
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    pltx.append(i)
    plty.append(r_value)
    if max_r<r_value:
        max_r = r_value
        max_pow = i

#plotting using plt
ax = plt
ax.scatter(pltx, plty)
ax.plot(pltx, plty)
ax.xlabel("Power to which calories are raised(i)")
ax.ylabel("R-Value")
ax.title("Proportionality between medals won and population times (calories)^i")
ax.savefig('plots/Importance-Of-Calories.jpg')
ax.show()

#as we increase the importance of calories the r value increases and converges aroung 0.6
x = pop*pow(cal, max_pow)
slope, intercept, r_value, p_value, std_err = linregress(x, y)
print("The r value after introduction of population in the picture is:", r_value)

#Consider a subset of countries and plot them:
medals = []
for i in range(len(x)):
    medals.append([z[i], y[i]])
netData = sorted(medals, key=lambda col: (col[1]), reverse=True)
netData = np.array(netData)

topMedals = netData[0:30]

#normalize medals w.r.t population
#asian games

asianGames = pd.read_csv('data/asianGames.csv')
asianArr = asianGames.to_numpy()
asianDict = {}

for ele in asianArr:
    asianDict[ele[0]] = ele[4]
    
graph = []

for key in asianDict:
    if key not in popDict.keys() or key not in gdpDict.keys():
        print(key)
    else:
        graph.append([key, int(gdpDict[key]), float(asianDict[key])*pow(10,6)/popDict[key]])

graph.append(['Chinese Taipei', gdpDict['Taiwan'], float(asianDict['Chinese Taipei'])*pow(10,6)/popDict['Taiwan']])
graph.append(['North Korea', 1300, float(asianDict['Korea, Dem. People’s Rep.'])*pow(10,6)/popDict['North Korea']])

graph[27][1] = 7612
graph[23][1] = 86118
graph[12][1] = 23443
graph[17][1] = 43103
graph[21][1] = 32373

graph = sorted(graph, key=lambda col: (col[1]))

graph = np.array(graph)

X = graph[:,1]
Z = graph[:,0]
X = X.astype('int')
X = np.log(X)
Y = graph[:,2]
Y = Y.astype('float')

list1 = ["Afghanistan","North Korea","Mongolia","Jordan","China","Kazakhstan","Mongolia","Bahrain","Hong Kong SAR, China","Macao SAR, China","Qatar","Singapore","Japan","India"]

print(type(X[0]))

Z = list(Z)

X_train = []
y_train = []
for i in range(len(X)):
    X_train.append([X[i]])
    y_train.append([y[i]])

m, b = np.polyfit(X, Y, 1)

plt.scatter(X, Y, color="orange")
plt.plot(X, m*X+b, color="black")
plt.ylabel('Medals per million population')
plt.xlabel('Log of GDP per capita')
for i, txt in enumerate(list1):
    index = Z.index(list1[i])
    plt.annotate(txt, (X[index], Y[index]))
plt.savefig('plots/Importance-Of-GDP-per-capita.jpg')
plt.show()