
import pandas as pd
import numpy as np
import sys 
import os
import numpy as np
from sklearn import linear_model, model_selection, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
##use drugs, drinks and smokes to predict whether someone dropped out or not 


def prep_KNNDF(raw_dataframe): 
    df_raw = pd.read_csv(raw_dataframe)
    drink_map = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5} ##ordinal map which can reflect intensity of drink frequency
    df_raw["drink_code"] = df_raw.drinks.map(drink_map)
    dropout_map = {"dropped out of college/university": 1, "dropped out of space camp": 1,
	"dropped out of two-year college": 1, "dropped out of masters program": 1,
	"dropped out of ph.d program": 1, "dropped out of high school": 1,
	"dropped out of law school": 1, "dropped out of med school": 1}
    df_raw["dropout_class"] = df_raw.education.map(dropout_map) ##recode education: not a drop out = 0, a drop out = 1
    df_raw['dropout_class'].fillna(0, inplace = True)
    drug_map = {"never": 0, "sometimes": 1, "often": 2}
    df_raw["drug_code"] = df_raw.drugs.map(drug_map)
    smoke_map = {"no": 0, "sometimes": 1, "when drinking": 1, "yes": 2, "trying to quit": 2}
    df_raw["smoke_code"] = df_raw.smokes.map(smoke_map)
    features_df = df_raw[['drink_code', 'drug_code', 'smoke_code', 'dropout_class']].copy()
    features_df = features_df.dropna() ##drop any row is there is a single NaN in it - reduces data from 60000 samples to 42000
    labels_df = features_df.filter(['dropout_class'], axis=1).reset_index(drop=True) ##add NaN filtered labels to its own dataframe
    features_df = features_df.drop(['dropout_class'], axis=1) ##remove filtered labels from feature df 
    return features_df, labels_df

	
def normalize_df(dataframe): 
    x = dataframe.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    feature_data_normalized = pd.DataFrame(x_scaled, columns=dataframe.columns)
	
    return feature_data_normalized


features_df, labels_df = prep_KNNDF(sys.argv[1])
feature_data_normalized = normalize_df(features_df) 
#print(labels_df.dropout_class.value_counts()) ##class distribution (40948 non-dropouts, 1547 dropouts) 


##set up arrays for model 
X = np.array(feature_data_normalized) 
y = np.array(labels_df) 
##set up classifier
log_reg = LogisticRegression()
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2) 
##get predicted labels
log_reg.fit(X_train, y_train.ravel())
y_prediction = log_reg.predict(X_test)
print(y_prediction, log_reg.score(X_test, y_test))  

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

for i in range(len(y_prediction)):
  #True Positives
  if y_train[i] == 1 and y_prediction[i] == 1:
    true_positives += 1
  #True Negatives
  if y_train[i] == 0 and y_prediction[i] == 0:
    true_negatives += 1
  #False Positives
  if y_train[i] == 0 and y_prediction[i] == 1:
    false_positives += 1
  #False Negatives
  if y_train[i] == 1 and y_prediction[i] == 0:
    false_negatives += 1
    
	
print(true_positives, true_negatives, false_positives, false_negatives) ##gives 0, 8198, 0, 301
accuracy = (true_positives + true_negatives) / len(y_prediction)
print("accuracy: ", accuracy)

if true_positives > 0: 
    recall = true_positives / (true_positives + false_negatives)
    print("recall: ", recall)
    precision = true_positives / (true_positives + false_positives)
    print("precision: ", precision)
    f_1 = 2*(precision*recall)/ (precision + recall) ##calculates harmonic mean rather than arithmetic mean 
    print("f1 score: ", f_1)
else: 
    recall = 0
    precision = 0
    f_1 = 0
    print("recall: ", recall)
    print("precision: ", precision)
    print("f1 score: ", f_1)

####k-folds test for determining optimal K value, and seeing if splitting samples is consistent throughout entire dataset#########################
# scores = cross_val_score(clf, X, y.ravel(), cv = 5, scoring = 'accuracy') 
# print(scores, "Mean out of sample scores: ", scores.mean()) 

# k_range = range(1, 31)
# k_scores = [] 
# for k in k_range: 
    # knn = neighbors.KNeighborsClassifier(n_neighbors = k) 
    # scores = cross_val_score(knn, X, y.ravel(), cv = 5, scoring = "accuracy") 
    # k_scores.append(scores.mean())
# print(k_scores) 
# plt.plot(k_range, k_scores) 
# plt.xlabel("K_Range") 
# plt.ylabel("Cross-Validated K-Scores")
# plt.show()



