import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#import dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

#split data into training + test set
x_training, x_test, y_training, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#training simple linear regression model
regress = LinearRegression()
regress.fit(x_training,y_training)

#predict test set results
salary_prediction = regress.predict(x_test)
print()