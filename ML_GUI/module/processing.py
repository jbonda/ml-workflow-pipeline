# Processing Module

from flask import flash, session
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class DMM():
    def split_data(self, test_size):
        """Method to split the data into training and testing subsets.."""
        if (self.data is not None):
            self.x = self.data[[self.selected_input_column]]
            self.y = self.data[self.selected_target_column]
            # Select input and target columns
            x_train, x_test, y_train, y_test = train_test_split(
                self.x, self.y, test_size=test_size, random_state=42
            )
            # Split the data
            self.x_train, self.x_test, self.y_train, self.y_test = (
                pd.DataFrame(x_train),
                pd.DataFrame(x_test),
                pd.DataFrame(y_train),
                pd.DataFrame(y_test),
            )  # Store training and testing data
            flash("Data split successfully!", "success")
            # Flash a success message
        else:
            flash("Please select input and target columns and upload data.", "danger")
            # Flash a message if no data is uploaded or columns are not selected

    def visualize_data(self, x, y, title):
        """Method to create a scatter plot for data visualization."""
        if x is not None and y is not None:
            # Plot Generation
            fig, axes = plt.subplots(figsize=(8, 6))
            axes.scatter(x, y)
            plt.show()
            axes.set_title(f"Scatter Plot - {title}")
            axes.set_xlabel(self.selected_input_column)
            axes.set_ylabel(self.selected_target_column)
            plt.tight_layout()

            return design_format()
        else:
            flash("Invalid data for visualization.", "danger")
            return None
            # Flash an error message if data is invalid and return None

    def scale_data(self, scaling_method):
        """Method to scale the data using different methods."""
        if self.x is not None and self.y is not None:
            # Choose the appropriate scaler based on the specified method.
            if scaling_method == "standard":
                scaler = StandardScaler()
            elif scaling_method == "min_max":
                scaler = MinMaxScaler()
            else:
                flash("Invalid scaling method specified.", "danger")
                return

            # Scale training & testing data, target variables, and convert to DataFrames.
            self.x_train_scaled = scaler.fit_transform(self.x_train)
            self.x_test_scaled = scaler.transform(self.x_test)

            self.y_train_scaled = scaler.fit_transform(self.y_train.values.reshape(-1,1))
            self.y_test_scaled = scaler.transform(self.y_test.values.reshape(-1,1))

            self.x_train_scaled = pd.DataFrame(self.x_train_scaled)
            self.x_test_scaled = pd.DataFrame(self.x_test_scaled)
            self.y_train_scaled = pd.DataFrame(self.y_train_scaled)
            self.y_test_scaled = pd.DataFrame(self.y_test_scaled)

            if scaling_method != "none":
                flash("Data scaled successfully!", "success")
                # Flash a success message if scaling is successful
        else:
            flash("Please select input and target columns and upload data.", "danger")
            # Flash a message if no data is uploaded or columns are not selected

def generate_results(x_train, y_train):
    """Train simple linear regression model, plot predictions, and the original data."""

    # Model Initialization
    regressor = SGDRegressor()
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_train)

    # Plotting Results
    plt.clf()
    fig, axes = plt.subplots(figsize=(8, 6))
    plt.plot(x_train, y_pred, '--', label='Predictions', alpha=0.5)
    plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
    plt.xlabel('Independent Variable')
    plt.ylabel('Dependent Variable')
    plt.legend()
    plt.title('Simple Linear Regression')
    plt.scatter(x_train, y_train)
    plt.show()

    return design_format()

def design_format():
    """Plot formatting and conversion to base64 encoding."""
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    graphic = base64.b64encode(image_png).decode()
    return graphic

