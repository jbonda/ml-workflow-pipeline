# Models Module

from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
import module.output as output


class ModelSelection:
    def simple_linear_regression(
        self,
        x_train_scaled,
        y_train_scaled,
        x_test_scaled,
        y_test_scaled,
        loss,
        penalty,
        alpha,
        l1_ratio,
        fit_intercept,
        iterations,
        tol,
        shuffle,
        verbose,
        epsilon,
        random_state,
        learning_rate,
        eta0,
        power_t,
        early_stopping,
        validation_fraction,
        n_iter_no_change,
        warm_start,
        average,
    ):
        """Train simple linear regression model, plot predictions, and the original data."""

        # Model Initialization
        regressor = SGDRegressor(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=iterations,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            epsilon=epsilon,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            warm_start=warm_start,
            average=average,
        )
        regressor.fit(
            x_train_scaled, y_train_scaled
        )  # Pass y_train_scaled to fit method
        y_pred = regressor.predict(x_test_scaled)
        self.y_pred = y_pred
        # print("y_pred: ", self.y_pred.shape)
        # print("x_test: ", self.x_test_scaled.shape)
        # y_pred.flatten()
        # Plotting Results
        plt.clf()
        fig, axes = plt.subplots(figsize=(8, 6))
        plt.plot(y_test_scaled, y_pred, "go", label="True data", alpha=0.5)
        plt.plot(x_test_scaled, y_pred, "--", label="Predictions", alpha=0.5)

        plt.xlabel("Independent Variable")
        plt.ylabel("Dependent Variable")
        plt.title("Simple Linear Regression")
        plt.legend()
        plt.show()

        return output.design_format()

    def polynomial_linear_regression():
        return None

    def logistic_regression():
        return None
