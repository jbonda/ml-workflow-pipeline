# Input Module

from flask import flash, session
import pandas as pd
import zipfile
from module.processing import DMM

class DataModelManager(DMM):
    """Class definition to manage the data model."""
    def __init__(self):
        """Class initialization attributes."""
        self.data = None  # Holds the dataset
        self.x = None  # Input features
        self.y = None  # Target variable
        self.columns = []  # List of column names
        self.selected_input_column = None  # Holds the name of the selected input column
        self.selected_target_column = None  # Holds the name of the selected target column
        self.x_train = None  # Training input features
        self.x_test = None  # Testing input features
        self.y_train = None  # Training target variable
        self.y_test = None  # Testing target variable
        self.x_scaled = None  # Scaled input features
        self.y_scaled = None  # Scaled target variable
        self.x_train_scaled = None  # Scaled training input features
        self.y_train_scaled = None  # Scaled training target variable
        self.x_test_scaled = None  # Scaled testing input features
        self.y_test_scaled = None  # Scaled testing target variable

    def load_data(self, file):
        """Method to load data from a file."""
        try:
            if file.filename.endswith(".csv"):
                # If the uploaded file is a CSV
                self.data = pd.read_csv(file, encoding='ISO-8859-1')
                # Read the CSV into a DataFrame
                self.fill_empty_columns()
                # Fill empty column names if they start with "Unnamed"
                self.columns = list(self.data.columns)  # Store the column names
                return True  # Indicate successful loading
            elif file.filename.endswith(".zip"):
                # If the uploaded file is a ZIP archive
                with zipfile.ZipFile(file, 'r') as zip_ref:
                    csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                    # Get a list of all CSV files in the archive
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
                return True  # Indicate successful loading
            else:
                flash("Please upload a CSV or ZIP file.", "danger")
        except Exception as e:
            flash(f"Error: {str(e)}", "danger")
        return False

    def fill_empty_columns(self):
        """Method to fill empty column names."""
        if not self.data.columns[0].startswith("Unnamed"):
            self.data.columns = [
                        f"Column {i}" for i in range(1, len(self.data.columns) + 1)
                    ]
            # If the column names don't start with "Unnamed", name them as "Column 1", "Column 2", etc.

    def remove_NaN_values(self):
        """Method to remove rows with NaN values."""
        if self.data is not None:
            if self.data.isnull().values.any():
                try:
                    self.data = self.data.dropna()
                    flash("NaN Values are removed successfully!", "success")
                    # Drop rows with NaN values and flash a success message
                except Exception as e:
                    flash(f"Error cleaning data: {str(e)}", "danger")
                    # Flash an error message if an exception occurs
            else:
                flash("No NaN values present in the uploaded file.")
                # Flash a message if no NaN values are found
        else:
            flash("Please upload a CSV file before using this function.", "danger")
            # Flash a message if no data is uploaded

    def remove_duplicates(self):
        """Method to remove duplicate rows."""
        if self.data is not None:
            if self.data.duplicated().any():
                try:
                    initial_shape = self.data.shape
                    self.data = self.data[~self.data.duplicated()]
                    final_shape = self.data.shape
                    flash(f"Removed {initial_shape[0] - final_shape[0]} duplicate row(s).", "success")
                    # Remove duplicates and flash a success message with the count of removed rows
                except Exception as e:
                    flash(f"Error removing duplicates: {str(e)}", "danger")
                    # Flash an error message if an exception occurs
            else:
                flash('No duplicate values present in the uploaded data file.', "info")
                # Flash a message if no duplicates are found
        else:
            flash("Please upload a CSV file before using this function.", "danger")
            # Flash a message if no data is uploaded

