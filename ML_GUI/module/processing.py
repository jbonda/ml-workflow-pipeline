# Processing Module

from flask import session
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def generate_results(X_train, y_train):
        """Train simple linear regression model, plot predictions, and the original data."""

        # Model Initialization
        regressor = SGDRegressor()
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_train)


        # Plotting Results
        plt.clf()
        fig, axes = plt.subplots(figsize=(8, 6))
        plt.plot(X_train, y_pred, '--', label='Predictions', alpha=0.5)
        plt.plot(X_train, y_train, 'go', label='True data', alpha=0.5)
        plt.xlabel('Independent Variable')
        plt.ylabel('Dependent Variable')
        plt.title('Simple Linear Regression')
        plt.scatter(X_train, y_train)
        plt.show()

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        graphic = base64.b64encode(image_png).decode()
        return graphic
