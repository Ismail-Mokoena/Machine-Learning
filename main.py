import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

#import dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


#taking care of missing data
"Replace missing values with an average"
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#encoding categorical data
"create a binary vector"
col_trans = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],remainder='passthrough')
x = np.array(col_trans.fit_transform(x))

#encode dependant variable
label_encode = LabelEncoder()
y = label_encode.fit_transform(y)