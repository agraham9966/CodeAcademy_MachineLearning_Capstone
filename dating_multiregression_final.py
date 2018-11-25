import pandas as pd
import numpy as np
import sys 
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import linear_model, preprocessing, neighbors, model_selection, metrics  
from sklearn.model_selection import train_test_split, cross_val_score
from matplotlib import pyplot as plt
##use drugs, drinks and smokes to predict whether someone dropped out or not 


def prep_MLregressionDF(raw_dataframe): 

    df_raw = pd.read_csv(sys.argv[1])
    features_df = df_raw.groupby('income').age.mean().reset_index()
    drug_map = {"never": 0, "sometimes": 1, "often": 2}
    df_raw["drug_code"] = df_raw.drugs.map(drug_map)
    drug_income = df_raw.groupby('income').drug_code.mean().reset_index()
    features_df['drugs'] = drug_income[['drug_code']]
    features_df = features_df[features_df.income != -1] 
	
    features_df['Income_Bracket'] = pd.cut(features_df['income'], 
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
    features_df["Income_Code"] = features_df.Income_Bracket.map(income_bracket_map)

    return features_df
	
	
features_df = prep_MLregressionDF(sys.argv[1])	

X = features_df[['Income_Code', 'drugs']]
y = features_df[['age']]

# # ##get predicted labels
mlr = LinearRegression()
model = mlr.fit(X, y)

y_predict = mlr.predict(X)
print(model.score(X, y))
print(model.coef_)

plt.plot(X[['Income_Code']], y_predict, '-')
plt.scatter(X[['Income_Code']], y, marker = "o")
plt.xlabel("Income Code")
plt.ylabel("Ages")
plt.title("Income Code vs. Predicted Age")
plt.show()

