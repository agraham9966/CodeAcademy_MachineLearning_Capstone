##https://stackoverflow.com/questions/13413590/how-to-drop-rows-of-pandas-dataframe-whose-value-in-certain-columns-is-nan    ## dealing with NA values 
import pandas as pd
import numpy as np
import sys 
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from matplotlib import pyplot as plt


def prep_regressionDF(raw_dataframe): 
    df_raw = pd.read_csv(raw_dataframe)
    age_income = df_raw.groupby('income').age.mean().reset_index()
    age_income = age_income[age_income.income != -1] ##remove column with '-1' because no clue what that is for... welfare??
    age_income['Income_Bracket'] = pd.cut(age_income['income'], 
	[0, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 100000, 150000, 250000, 500000, 1000000], 
	labels=['0-20000', '20001-30000', '30001-40000', 
	'40001-50000', '50001-60000', '60001-70000', 
	'70001-80000', '80001-100000', '100001-150000', 
	'150001-250000', '250001-500000', '500001-1000000'])
    income_bracket_map = {
        "0-20000": 0,
        "20001-30000": 1,
        "30001-40000": 2,
        "40001-50000": 3,    
        "50001-60000": 4,
        "60001-70000": 5,
        "70001-80000": 6,
        "80001-100000": 7,
        "100001-150000": 8,
        "150001-250000": 9,
        "250001-500000": 10,
        "500001-1000000": 11
    }
    age_income["Income_Code"] = age_income.Income_Bracket.map(income_bracket_map)
    return age_income 

#######run a quick regression with the prepped dataframe (Mean Age vs. Income Bracket) 	
df1 = prep_regressionDF(sys.argv[1])
df1 = df1[df1.income != 20000] ##dropped columns which were skewing linear regression 
df1 = df1[df1.income != 500000]
df1 = df1[df1.income != 1000000]


print(df1)

X = df1[['Income_Code']]
X.values.reshape(-1, 1) 
y = df1[['age']] 
y.values.reshape(-1, 1) 

regr = linear_model.LinearRegression()
regr.fit(X, y)
print(regr.score(X, y))

y_predict = regr.predict(X)

plt.plot(X, y_predict, '-') 
plt.scatter(X, y) ##poor linear fit 
plt.xlabel("Income_Code")
plt.ylabel("Mean Age")
plt.title("Mean Age Vs. Income Bracket") 
plt.show()





####excess things to call when needed ##########

#print(df_raw.offspring.value_counts())  ##shows histogram of all possible items in a column 
#print(df_raw.columns.values)  ##shows all the column names 

# plt.hist(df_raw.religion, bins=20)
# plt.xlabel("Age")
# plt.ylabel("Frequency")
# plt.xlim(16, 80)
# plt.show()