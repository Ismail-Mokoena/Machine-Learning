import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split



#data preprocessing
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:-1].values
#last column 1010
y = dataset.iloc[:,-1].values

#Encode categorical data
label_encode = LabelEncoder()
#gender column
x[:, 2] = label_encode.fit_transform(x[:, 2])
#encode geo
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[1])],remainder='passthrough')
x = np.array(ct.fit_transform(x))
#split dataset training/test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 0)
#feature scaling
s_scaler = StandardScaler()
x_train = s_scaler.fit_transform(x_train)
x_test = s_scaler.fit_transform(x_test)

#Initialize ANN - brain
"sequence of layers"
ann = tf.keras.models.Sequential()
#hidden layer
for i in range(2):
  "two layer"
  ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

#output layer - binary need 1 output neuron
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

#Train Ann
#compile Ann
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#Train
ann.fit(x_train, y_train,batch_size=32,epochs=100)

