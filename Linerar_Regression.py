# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:30:37 2024

@author: iMi
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the data set from Desktop
dataset = pd.read_csv("Salary_DataSet.csv")
X = dataset.iloc[:, :-1].values.reshape(-1, 1)  # Reshape to 2D
Y = dataset.iloc[:, 1].values

# Training and Testing Data (divide the data into two parts)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

# Regression
reg = LinearRegression()
reg.fit(X_train, Y_train)
Y_predict = reg.predict(X_test)

# Visualization
plt.scatter(X_train, Y_train, color='red', label='Training data')
plt.scatter(X_test, Y_test, color='blue', label='Test data')
plt.plot(X_train, reg.predict(X_train), color='green', label='Regression line')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()
