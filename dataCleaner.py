
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
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import warnings

df = pd.read_csv('/Users/jsandoval/Desktop/Coding/Airline Delay/Airlines.csv')

df = df.drop(['DayOfWeek', 'Time'], axis=1)

df['Delay'].plot(kind='hist', bins=50, figsize=(12, 12))
# plt.show()

# REDUCING DATA SIZE

df_delay = df[df['Delay'] == 1]

df_nodelay = df[df['Delay'] == 0]

df_nodelay = df_nodelay.head(100000)

df = pd.concat([df_delay, df_nodelay], axis=0)

df['Delay'].plot(kind='hist', bins=50, figsize=(12, 12))
# plt.show()


# ENCODING

# Airline
df['Airline'] = df['Airline'].astype('category')
Airline_encode = preprocessing.LabelEncoder()
df['Airline'] = Airline_encode.fit_transform(
    df.Airline)

# AirportFrom
df['AirportFrom'] = df['AirportFrom'].astype('category')
AirportFrom_encode = preprocessing.LabelEncoder()
df['AirportFrom'] = AirportFrom_encode.fit_transform(
    df.AirportFrom)


# AirportTo
df['AirportTo'] = df['AirportTo'].astype('category')
AirportTo_encode = preprocessing.LabelEncoder()
df['AirportTo'] = AirportTo_encode.fit_transform(
    df.AirportTo)


# One hot encoder

# Airline
Airline_one_hot = OneHotEncoder()
Airline_one_hot_encode = Airline_one_hot.fit_transform(
    df.Airline.values.reshape(-1, 1)).toarray()

ohe_variable = pd.DataFrame(Airline_one_hot_encode, columns=[
                            "Airline_"+str(int(i)) for i in range(Airline_one_hot_encode.shape[1])])
df = pd.concat([df, ohe_variable], axis=1)

# AirportFrom
AirportFrom_one_hot = OneHotEncoder()
AirportFrom_one_hot_encode = AirportFrom_one_hot.fit_transform(
    df.AirportFrom.values.reshape(-1, 1)).toarray()

ohe_variable = pd.DataFrame(AirportFrom_one_hot_encode, columns=[
                            "AirportFrom_"+str(int(i)) for i in range(AirportFrom_one_hot_encode.shape[1])])
df = pd.concat([df, ohe_variable], axis=1)

# AirportTo
AirportTo_one_hot = OneHotEncoder()
AirportTo_one_hot_encode = AirportTo_one_hot.fit_transform(
    df.AirportTo.values.reshape(-1, 1)).toarray()

ohe_variable = pd.DataFrame(AirportTo_one_hot_encode, columns=[
                            "AirportTo_"+str(int(i)) for i in range(AirportTo_one_hot_encode.shape[1])])
df = pd.concat([df, ohe_variable], axis=1)


# dropping the original variables
df = df.drop(['Airline', 'AirportTo',
             'AirportFrom'], axis=1)

print(df.head())
print(df.shape)
df = df.fillna(0)
# print(df.isnull().any())


#df.to_csv('/Users/jsandoval/Desktop/Coding/Airline Delay/clean_data.csv')
