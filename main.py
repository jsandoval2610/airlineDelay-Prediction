import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
import warnings
import pdb

df = pd.read_csv(
    '/Users/jsandoval/Desktop/Coding/Airline Delay/clean_data.csv')

features = df.drop('Delay', axis=1).values
target = df['Delay'].values


X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2)

warnings.filterwarnings('ignore')
'''
lasso_regression = Lasso()

# Using GridSearchCV to search for the best parameter

grid = GridSearchCV(lasso_regression, {
                    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]})
grid.fit(X_train, y_train)

# Print out the best parameter

print("The most optimal value of alpha is:", grid.best_params_)


'''

lasso_regression = Lasso(alpha=0.0001)

lasso_regression.fit(X_train, y_train)

print(lasso_regression.score(X_test, y_test))
