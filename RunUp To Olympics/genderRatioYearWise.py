#gender ratio of games throughout the year
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
# import re
from sklearn.svm import SVR

def fileSaver(answer, colNames, filename):
    answer = pd.DataFrame(answer)
    answer.columns = colNames
    answer.to_csv(filename, index=False) 

olmpData = pd.read_csv('data/athlete_events.csv')
olmpArr = olmpData.to_numpy()

yearWiseNames = {}

#ratio of females/males in summer olympics
for i in range(len(olmpArr)):
    games = olmpArr[i][8]
    splitt = games.split()
    year = int(splitt[0])
    if splitt[1]=="Winter":
        continue
    gender = olmpArr[i][2]
    yearWiseNames[year] = {"M": 0, "F":0}

#ratio of females/males in summer olympics
for i in range(len(olmpArr)):
    games = olmpArr[i][8]
    splitt = games.split()
    year = int(splitt[0])
    if splitt[1]=="Winter":
        continue
    gender = olmpArr[i][2]
    yearWiseNames[year][gender]+=1

y = []
x = []
for key in sorted(yearWiseNames):
    x.append(key)
    rat = yearWiseNames[key]['F']/yearWiseNames[key]['M']
    y.append(rat)

#plotting using plt
ax = plt
ax.scatter(x, y)
ax.xlabel("Year")
ax.ylabel("Ratio of female to male participants")

ax.savefig('plots/genderRatio.jpg')
ax.show()

#code to predict 2020's ratio
from sklearn.linear_model import Ridge

X = []
for ele in x:
    X.append([ele])

# Fit regression model
svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
# svr_lin = SVR(kernel="linear", C=100, gamma="auto")
# svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)

print(1)
model = svr_rbf
model.fit(X, y)
print(2)
model.predict([[2020]])
X_pred = []
for x in X:
    X_pred.append(x)
X_pred.append([2020])
y_pred = model.predict(X_pred)

score = 0
y = np.array(y)
for i in range(len(y)):
    score += (y[i]-y_pred[i])**2

print(score)

print(1)
model = Ridge()
model.fit(X, y)
print(2)
model.predict([[2020]])
X_pred = []
for x in X:
    X_pred.append(x)
X_pred.append([2020])
y_pred = model.predict(X_pred)

score = 0
y = np.array(y)
for i in range(len(y)):
    score += (y[i]-y_pred[i])**2

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree = 3)
X_poly = poly.fit_transform(X)
 
poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)

print(score)

y_pred = lin2.predict(X_poly)
y_pred = np.ravel(y_pred)

score = 0
y = np.array(y)
for i in range(len(y)):
    score += (y[i]-y_pred[i])**2
print(score)

plt.scatter(X, y, color = "red")
plt.plot(X, y_pred, color = "green")
plt.title("Polynomial Regression of degree-4")
plt.xlabel("Year")
plt.ylabel("Ratio of female to male participants")
plt.savefig("plots/regressionOnGenderRatio.jpg")

ratio = 5386/5704

print("Ratio in 2020:", ratio)
print("predicted ratio in 2020: ", lin2.predict(poly.fit_transform([[2020]])))