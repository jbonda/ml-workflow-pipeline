# Processing Module

import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
import matplotlib.pyplot as plt

def train_model(X_train, y_train):
    """Simple linear regression model training."""

    # Create a LinearRegression object
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Plot the regression line
    plt.scatter(X_train, y_train)
    plt.plot(X_train, model.predict(X_train), color='red')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

    # Return the trained model
    return model


"Example Output Values (GUI)"
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([[6], [7], [8], [9], [10]])

train_model(X_train, y_train)
