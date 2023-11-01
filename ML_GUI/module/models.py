# Models Module

from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
import module.output as output

class ModelSelection():
    def simple_linear_regression(self, x_train, y_train):
        """Train simple linear regression model, plot predictions, and the original data."""

        # Model Initialization
        regressor = SGDRegressor()
        regressor.fit(x_train, y_train)
        y_pred = regressor.predict(self.x_test)
        self.y_pred = y_pred
        print("y_pred: ", self.y_pred.shape)
        print("x_test: ", self.x_test.shape)

        # Plotting Results
        plt.clf()
        fig, axes = plt.subplots(figsize=(8, 6))
        plt.plot(self.x_test, self.y_test,  'go', label='True data', alpha=0.5)
        plt.plot(self.x_test, self.y_pred, '--', label='Predictions', alpha=0.5)

        plt.xlabel('Independent Variable')
        plt.ylabel('Dependent Variable')
        plt.title('Simple Linear Regression')
        plt.legend()
        plt.show()

        return output.design_format()

    def polynomial_linear_regression():
        return None

    def logistic_regression():
        return None
