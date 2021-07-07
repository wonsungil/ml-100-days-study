import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    dataset = pd.read_csv("dataset/50_Startups.csv")

    X = dataset.iloc[ : , :-1].values
    Y = dataset.iloc[ : , 4].values

    labelencoder = LabelEncoder()
    X[ : ,3] = labelencoder.fit_transform(X[ : ,3])

    ct = ColumnTransformer([('State', OneHotEncoder(), [3])], remainder = 'passthrough')
    X = ct.fit_transform(X)

    X = X[:, 1:]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)

    print(X_train)
    print(X_train[: , 0])
    print(X_train[: , 1])

    Y_pred = regressor.predict(X_test)
