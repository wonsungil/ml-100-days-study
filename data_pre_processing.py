import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    dataset = pd.read_csv("dataset/Data.csv")
    X = dataset.iloc[ :, :-1].values
    Y = dataset.iloc[ : , 3].values

    print("-----------------------------------------------------------------------------------------------------------",
      "X : {}".format(X),
      "Y : {}".format(Y), sep="\n")

    imputer = SimpleImputer(missing_values= np.nan, strategy= "mean")
    imputer = imputer.fit(X[ :, 1:3])
    X[ : , 1:3] = imputer.transform(X[ : , 1:3])

    print("-----------------------------------------------------------------------------------------------------------",
      "X : {}".format(X),
      "Y : {}".format(Y), sep="\n")

    ct = ColumnTransformer([('Country', OneHotEncoder(), [0])], remainder= 'passthrough')
    X = ct.fit_transform(X)

    print("-----------------------------------------------------------------------------------------------------------",
          "X : {}".format(X),
          "Y : {}".format(Y), sep="\n")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)

    print("-----------------------------------------------------------------------------------------------------------",
          "X_train : {}".format(X_train),
          "X_test : {}".format(X_test), sep="\n")

    stdSc = StandardScaler()
    X_train_std = stdSc.fit_transform(X_train)
    X_test_std = stdSc.fit_transform(X_test)

    print("#StandardScaler#",
          "X_train : {}".format(X_train_std),
          "X_test : {}".format(X_test_std), sep="\n")

    mmSc = MinMaxScaler()
    X_train_mm = mmSc.fit_transform(X_train)
    X_test_mm = mmSc.fit_transform(X_test)

    print("#MinMaxScaler#",
          "X_train : {}".format(X_train_mm),
          "X_test : {}".format(X_test_mm), sep="\n")

    maSc = MaxAbsScaler()
    X_train_ma = maSc.fit_transform(X_train)
    X_test_ma = maSc.fit_transform(X_test)

    print("#MaxAbsScaler#",
          "X_train : {}".format(X_train_ma),
          "X_test : {}".format(X_test_ma), sep="\n")

    rbSc = RobustScaler()
    X_train_rb = rbSc.fit_transform(X_train)
    X_test_rb = rbSc.fit_transform(X_test)

    print("#RobustScaler#",
          "X_train : {}".format(X_train_rb),
          "X_test : {}".format(X_test_rb), sep="\n")