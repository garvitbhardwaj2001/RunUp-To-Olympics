import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json

#k-means on weight and height of athletes
data = pd.read_csv('data/athlete_events.csv')
dataArr = data.to_numpy()

ht = dataArr[:,4]
wt = dataArr[:,5]
sport = dataArr[:,12]

X = []
lab = []

def isNan(val):
    return val!=val

for i in range(len(ht)):
    if isNan(ht[i])==False and isNan(wt[i])==False and dataArr[i][9]==2016:
        X.append([ht[i], wt[i]])
        lab.append(sport[i])

X = np.array(X)

plt.scatter(X[:,0], X[:,1])

kmeans = KMeans(n_clusters=6, random_state=0).fit(X)
label = kmeans.fit_predict(X)

colors = ['Red', 'Blue', 'Yellow', 'Orange', 'Green', 'Black', 'Violet']
col = []
for i in range(len(label)):
    col.append(colors[int(label[i])])

legends = []
for i in range(len(label)):
    legends.append("cluster-id-" + str(label[i]))

plt.scatter(X[:,0], X[:,1], c=col, label = label, alpha = 0.6, s=10)
plt.xlabel('Height in cms')
plt.ylabel('Weight in kgs')
plt.savefig('plots/heigh-weight-clusters.jpg')
plt.show()

#for each cluster find top sports:
top = {}
for i in range(len(label)):
    top[label[i]] = {}

for i in range(len(label)):
    top[label[i]][lab[i]]=0

for i in range(len(label)):
    top[label[i]][lab[i]]+=1
    
sportsWise = {}
for game in lab:
    sportsWise[game]=0
for game in lab:
    sportsWise[game]+=1    

top3sports = {}

for key in top:
    data = top[key]
    total = 0
    x = []
    for ele in data:
        total+=data[ele]
        x.append([ele, data[ele]])
    
    temp = {}
    for i in range(len(x)):
        x[i][1] = x[i][1]*100/sportsWise[x[i][0]]
        
    x = sorted(x, key=lambda col: (col[1]), reverse=True)
    
    for i in range(0, 3):
        temp[x[i][0]] = x[i][1]
    # temp[x[0][0]] = x[0][1]*100/sportsWise[x[0][0]]
    # temp[x[1][0]] = x[1][1]*100/sportsWise[game]
    # temp[x[2][0]] = x[2][1]*100/sportsWise[game]
    # temp[x[3][0]] = x[3][1]*100/sportsWise[game]
    # temp[x[4][0]] = x[4][1]*100/sportsWise[game]
    # temp[x[5][0]] = x[5][1]*100/sportsWise[game]
    top3sports[np.int(key)] = temp
    print(key)

with open('json/clusterWiseGames.json', 'a') as fp:
    fp.truncate(0)
    json.dump(top3sports, fp, indent=2)