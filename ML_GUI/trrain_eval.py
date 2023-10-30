def generate_results(x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled):
        # Model Initialization
        regressor = SGDRegressor()
        regressor.fit(x_train_scaled, y_train_scaled)
        y_pred_train = regressor.predict(x_train_scaled)
        y_pred_test = regressor.predict(x_test_scaled)

        # Create a figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(8, 12))
        
        axes[0].scatter(y_train_scaled, y_pred_train, 'bo', label='Predictions')
        axes[0].plot(y_train_scaled, y_train_scaled, 'r--', label='True Data', alpha=0.5)
        axes[0].set_xlabel('True Values')
        axes[0].set_ylabel('Predictions')
        axes[0].legend()
        axes[0].set_title('Training Data')

        axes[1].scatter(y_test_scaled, y_pred_test, 'bo', label='Predictions')
        axes[1].plot(y_test_scaled, y_test_scaled, 'r--', label='True Data', alpha=0.5)
        axes[1].set_xlabel('True Values')
        axes[1].set_ylabel('Predictions')
        axes[1].legend()
        axes[1].set_title('Test Data')

        plt.tight_layout()
        plt.show()

        return design_format()

    def calculate_rmse(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return rmse

    def calculate_accuracy(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy
