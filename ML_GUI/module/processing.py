# Processing Module

from flask import flash, session
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from module.models import ModelSelection
from module.output import design_format
import matplotlib.pyplot as plt


class DMM(ModelSelection):
    def split_data(self, test_size):
        """Method to split the data into training and testing subsets.."""
        if self.data is not None:
            # Take multiple and store them for the scaling method.
            if isinstance(self.selected_input_column, list):
                self.x = self.data[self.selected_input_column]
            else:
                self.x = self.data[[self.selected_input_column]]
            self.y = self.data[self.selected_target_column]

            # Check if duplicates are removed and NaN values are successfully dealt with.
            if (self.x.isnull().values.any() or self.y.isnull().values.any()) and (
                self.x.duplicated().any() or self.y.duplicated().any()
            ):
                flash(
                    "Please complete the preprocessing steps before splitting the data.",
                    "danger",
                )
                return None, None, None, None
            if self.x.isnull().values.any() or self.y.isnull().values.any():
                flash("Please remove NaN values before splitting the data.", "danger")
                return None, None, None, None
            if self.x.duplicated().any() or self.y.duplicated().any():
                flash(
                    "Please remove duplicate values before splitting the data.",
                    "danger",
                )
                return None, None, None, None
            # If input column and target column names are the same, flash an error message.
            if self.selected_input_column == self.selected_target_column:
                flash("Input column and target column cannot be the same.", "danger")
                return None, None, None, None
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
            print("x_train: ", self.x_train.shape)
            print("x_test: ", self.x_test.shape)
            flash("Data split successfully!", "success")
            flash("Train data shape: " + str(self.x_train.shape))
            flash("Test data shape: " + str(self.x_test.shape))
            # Flash a success message
            return self.x_train, self.x_test, self.y_train, self.y_test
        else:
            flash("Please select input and target columns and upload data.", "danger")
            # Flash a message if no data is uploaded or columns are not selected
            return None, None, None, None

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

            self.y_train_scaled = scaler.fit_transform(
                self.y_train.values.reshape(-1, 1)
            )
            self.y_test_scaled = scaler.transform(self.y_test.values.reshape(-1, 1))

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
