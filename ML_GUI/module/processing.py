# Processing Module

from flask import session
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def train_model(X_train, y_train):
        """Train simple linear regression model."""

        # Take input of trained data.

        # Create a LinearRegression object
        model = LinearRegression()

        # Fit the model to the training data
        model.fit(X_train, y_train)

        # Predict on the test data (X_test)
        # y_pred = regressor.predict(self.X_test)

        # Plot the regression line
        plt.clf()
        plt.title('Simple Linear Regression')
        fig, axes = plt.subplots(figsize=(8, 6))
        plt.plot(X_train, model.predict(X_train), color='red', linestyle='dashed', label='Regression Line')
        plt.legend(loc='upper right')
        plt.xlabel('Independent variable (true data)')
        plt.ylabel('Dependent variable (predicted data)')
        plt.title('Simple Linear Regression')
        plt.legend(loc='best')
        plt.scatter(X_train, y_train)
        plt.show()

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        graphic = base64.b64encode(image_png).decode()
        return graphic
