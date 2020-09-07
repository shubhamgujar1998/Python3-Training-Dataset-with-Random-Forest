import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz

#read from the csv file and return a Pandas DataFrame.
nba = pd.read_csv('NBAstats.csv')

# print the column names
original_headers = list(nba.columns.values)
print(original_headers)

#print the first three rows.
print(nba[0:3])

# "Position (pos)" is the class attribute we are predicting. 
class_column = 'Pos'

#The dataset contains attributes such as player name and team name. 
#We know that they are not useful for classification and thus do not 
#include them as features. 
# ORIGINAL DATASET
feature_columns = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', \
    '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', \
    'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PS/G']
# feature_columns = ['G', 'GS', 'MP', 'FG%', '3PA', \
#     '3P%', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', \
#     'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PS/G']
#Pandas DataFrame allows you to select columns. 
#We use column selection to split the data into features and class. 
nba_feature = nba[feature_columns]
nba_class = nba[class_column]

print(nba_feature[0:3])
print(list(nba_class[0:3]))

train_feature, test_feature, train_class, test_class = \
    train_test_split(nba_feature, nba_class, stratify=nba_class, \
    train_size=0.75, test_size=0.25)

training_accuracy = []
test_accuracy = []

randomF = RandomForestClassifier().fit(train_feature, train_class)
print("Training set score: {:.3f}".format(randomF.score(train_feature, train_class)))
print("Random Forest Tree Classifier Test set score: {:.3f}".format(randomF.score(test_feature, test_class)))

prediction = randomF.predict(test_feature)
print("Confusion matrix below:\n")
print(pd.crosstab(test_class,prediction, rownames=['Player position'], colnames=['Predicted player position'],margins = 'True' ))
print("\n")
scores = cross_val_score(randomF, nba_feature, nba_class, cv=10)
print("Cross-validation scores with 10-fold startified:\n{}".format(scores))
print("\nAverage cross-validation for Random Forest Classifier score: {:.2f}".format(scores.mean()))