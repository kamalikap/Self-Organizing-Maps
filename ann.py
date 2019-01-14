#Artificial Neural Network


#Part 1 - Data Preprocessing

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:, -1].values

# Encoding categorical data- changing the names into numbers for easier calculation.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1= LabelEncoder()
X[:, 1]= labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2= LabelEncoder()
X[:, 2]= labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X= X[:, 1:]

#Splitting dataseet into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#Part 2 - Let's make ANN

import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation = 'relu' , input_dim = 11))


#Adding the second hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation = 'relu' ))

#Adding the output layer, (suftmax-if depenent variable has more than two categories)
classifier.add(Dense(output_dim=1, init='uniform', activation = 'sigmoid'))


#Compiling the ANN
classifier.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

#Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch =100)


#Part 3 - Making the predictions and the first hidden layer


#Predicting the test set results.
y_pred = classifier.predict(X_test)
y_pred = (y_pred> 0.52)


#Predicting  a single new observation
"""
Predict if the customer with the following information will leave the bank:
    Geography: France
    Credit Score: 600
    Gender: Male
    Age: 40
    Tenure: 3
    Balance:60000
    Number of Products:2
    Has Credit Card:Yes
    Estimated Salary: 50000
"""


new_pred = classifier.predict(sc_X.transform(np.array([[0.0,0,600,1, 40,3,60000, 2, 1,1, 50000]])))
new_pred = (new_pred >0.52)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)










