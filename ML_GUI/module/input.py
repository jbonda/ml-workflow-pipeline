# Input Module

import os
import secrets
import numpy as np
from flask import flash, session
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import zipfile

class DataModelManager:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.columns = []
        self.selected_input_column = None
        self.selected_target_column = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_scaled = None
        self.y_scaled = None

    def load_data(self, file):
        try:
            if file.filename.endswith(".csv"):
                self.data = pd.read_csv(file, encoding='ISO-8859-1')
                self.fill_empty_columns()
                self.columns = list(self.data.columns)  # Store the column names
                return True
            elif file.filename.endswith(".zip"):
                with zipfile.ZipFile(file, 'r') as zip_ref:
                    csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                    if len(csv_files) == 0:
                        flash("No CSV files found in the ZIP archive.", "danger")
                    elif len(csv_files) > 1:
                        first_file = pd.read_csv(zip_ref.open(csv_files[0]), encoding='ISO-8859-1')
                        self.data = first_file
                        self.fill_empty_columns()
                        self.columns = list(first_file.columns)  # Store the column names
                        for csv_file in csv_files[1:]:
                            data = pd.read_csv(zip_ref.open(csv_file), encoding='ISO-8859-1')
                            if len(first_file.columns) != len(data.columns) or first_file.columns[0] != data.columns[0]:
                                flash("Names/number of columns in the uploaded file(s) does not match in the ZIP archive.", "warning")
                                break
                            else:
                                if not first_file.equals(data):
                                    flash(f"File {csv_file} is different from the first file in the ZIP archive.", "warning")
                                    break
                        else:
                            flash("All files in the ZIP archive are identical.", "success")
                    else:
                        self.data = pd.read_csv(zip_ref.open(csv_files[0]), encoding='ISO-8859-1')
                        self.fill_empty_columns()
                        self.columns = list(self.data.columns)  # Store the column names
                return True
            else:
                flash("Please upload a CSV or ZIP file.", "danger")
        except Exception as e:
            flash(f"Error: {str(e)}", "danger")
        return False

    def fill_empty_columns(self):
        if not self.data.columns[0].startswith("Unnamed"):
            self.data.columns = [
                        f"Column {i}" for i in range(1, len(self.data.columns) + 1)
                    ]

    def remove_NaN_values(self):
        if self.data is not None:
            if self.data.isnull().values.any():
                try:
                    self.data = self.data.dropna()
                    flash("NaN Values are removed successfully!", "success")
                except Exception as e:
                    flash(f"Error cleaning data: {str(e)}", "danger")
            else:
                flash("No NaN values present in the uploaded file.")
        else:
            flash("No NaN values present in the uploaded file!")
        return

    def remove_duplicates(self):
        if self.data is not None:
            if self.data.duplicated().any():
                try:
                    initial_shape = self.data.shape
                    self.data = self.data[~self.data.duplicated()]
                    final_shape = self.data.shape
                    flash(f"Removed {initial_shape[0] - final_shape[0]} duplicate row(s).", "success")
                except Exception as e:
                    flash(f"Error removing duplicates: {str(e)}", "danger")
            else:
                flash('No duplicate values present in the uploaded data file.', "info")
        else:
            flash("Please upload a CSV file before using this function.", "danger")
        return

    def split_data(self, test_size):
        if (
            self.data is not None
            and self.selected_input_column
            and self.selected_target_column
        ):
            self.X = self.data[[self.selected_input_column]]
            self.y = self.data[self.selected_target_column]
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=42
            )
            self.X_train, self.X_test, self.y_train, self.y_test = (
                X_train,
                X_test,
                y_train,
                y_test,
            )  # Store training and testing data
            flash("Data split successfully!", "success")
        else:
            flash("Please select input and target columns and upload data.", "danger")

    def visualize_data(self, X, y, title):
        if X is not None and y is not None:
            fig, axes = plt.subplots(figsize=(8, 6))
            axes.scatter(X, y)
            plt.show()
            axes.set_title(f"Scatter Plot - {title}")
            axes.set_xlabel(self.selected_input_column)
            axes.set_ylabel(self.selected_target_column)
            plt.tight_layout()

            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()

            graphic = base64.b64encode(image_png).decode()
            return graphic
        else:
            flash("Invalid data for visualization.", "danger")
            return None

    def scale_data(self, input_scaling_method, target_scaling_method):
        if self.X is not None and self.y is not None:
            # Perform scaling based on selected methods for input data
            if input_scaling_method == "standard":
                input_scaler = StandardScaler()
                self.X_scaled = input_scaler.fit_transform(self.X)
                print(self.X_scaled[:5])
            elif input_scaling_method == "min_max":
                input_scaler = MinMaxScaler()
                self.X_scaled = input_scaler.fit_transform(self.X)
                print(self.X_scaled[:5])
            elif input_scaling_method == "robust":
                input_scaler = RobustScaler()
                self.X_scaled = input_scaler.fit_transform(self.X)
                print(self.X_scaled[:5])

            # Perform scaling based on selected methods for target data
            if target_scaling_method == "standard":
                target_scaler = StandardScaler()
                self.y_scaled = target_scaler.fit_transform(
                    self.y.values.reshape(-1, 1)
                )
                print(self.y_scaled[:5])
            elif target_scaling_method == "min_max":
                target_scaler = MinMaxScaler()
                self.y_scaled = target_scaler.fit_transform(
                    self.y.values.reshape(-1, 1)
                )
                print(self.y_scaled[:5])
            elif target_scaling_method == "robust":
                target_scaler = RobustScaler()
                self.y_scaled = target_scaler.fit_transform(
                    self.y.values.reshape(-1, 1)
                )
                print(self.y_scaled[:5])
            flash("Data scaled successfully!", "success")
