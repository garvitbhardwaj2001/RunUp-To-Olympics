import pandas as pd 
import numpy as np
# import matplotlib.pyplot as plt
# import re

def fileSaver(answer, colNames, filename):
    answer = pd.DataFrame(answer)
    answer.columns = colNames
    answer.to_csv(filename, index=False) 

#col 2 has gold medals, 3 has silver medals and 4th has bronze medals
df = pd.read_csv('data/table-1.csv')
dataArr = df.to_numpy()
dataArr = dataArr[:-4,0:6]
newArr = []
for i in range(1, len(dataArr)):
    # splitt = np.array(dataArr[i][0].split())
    # name = splitt[0:len(splitt)-1]
    # code = splitt[-1]
    # print(name, code)
    splitt = np.array(dataArr[i][0].split('('))
    name = splitt[0].replace(u'\xa0', u'')
    dig_3 = splitt[1].replace(')','')
    code = dig_3[0:3]
    newArr.append([name, code, dataArr[i][1], dataArr[i][2], dataArr[i][3], dataArr[i][4], dataArr[i][5]])

#save the newArr
colNames = ["country","code","number-of-times-participated","total-gold-medals","total-silver-medals","total-bronze-medals","total-medals"]
fileSaver(newArr, colNames, "data/medalsTally.csv")