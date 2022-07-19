#Multuple linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


#Encoding the categorical data
#Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,3]=labelencoder_x.fit_transform(x[:,3])
#onehotencoder=OneHotEncoder(categorical_features=[3])
#x=onehotencoder.fit_transform(x).toarray()
ct=ColumnTransformer([("State",OneHotEncoder(),[3])],remainder='passthrough')
x=ct.fit_transform(x)

#Avoiding the dummy variable trap
x=x[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

#Fitting multiple linearregression o the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#Predicting the test set result
y_pred=regressor.predict(x_test)

plt.scatter(x_test,y_test,color='r')
plt.scatter(x_test,y_pred,color='g')
plt.plot(x_train,regressor.predict(x_train),color='b')
plt.title("Salary vs Experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
