import os
import secrets
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
from io import BytesIO
import random
import base64
import matplotlib

matplotlib.use("Agg")

app = Flask(__name__)
app.secret_key = secrets.token_hex(24)


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
                self.data = pd.read_csv(file)
                self.columns = list(self.data.columns)  # Store the column names
                return True
            else:
                flash("Please upload a CSV file.", "danger")
        except Exception as e:
            flash(f"Error: {str(e)}", "danger")
        return False

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
            elif input_scaling_method == "min_max":
                input_scaler = MinMaxScaler()
                self.X_scaled = input_scaler.fit_transform(self.X)
            elif input_scaling_method == "robust":
                input_scaler = RobustScaler()
                self.X_scaled = input_scaler.fit_transform(self.X)

            # Perform scaling based on selected methods for target data
            if target_scaling_method == "standard":
                target_scaler = StandardScaler()
                self.y_scaled = target_scaler.fit_transform(
                    self.y.values.reshape(-1, 1)
                )
            elif target_scaling_method == "min_max":
                target_scaler = MinMaxScaler()
                self.y_scaled = target_scaler.fit_transform(
                    self.y.values.reshape(-1, 1)
                )
            elif target_scaling_method == "robust":
                target_scaler = RobustScaler()
                self.y_scaled = target_scaler.fit_transform(
                    self.y.values.reshape(-1, 1)
                )

            flash("Data scaled successfully!", "success")


data_manager = DataModelManager()


@app.route("/")
def index():
    if "tab_id" not in session:
        session["tab_id"] = secrets.token_hex(24)
        session.permanent = False
    return render_template("index.html", columns=data_manager.columns)


@app.route("/upload", methods=["POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            if data_manager.load_data(file):
                flash("File uploaded successfully!", "success")
    return redirect(url_for("index"))


@app.route("/split", methods=["POST"])
def split_data():
    test_size = float(request.form["test_size"])
    data_manager.selected_input_column = request.form["input_column"]
    data_manager.selected_target_column = request.form["target_column"]
    data_manager.split_data(test_size)
    return redirect(url_for("index"))


@app.route("/visualization")
def visualization():
    return render_template(
        "visualization.html", graphic=None, columns=data_manager.columns
    )


@app.route("/visualize_whole", methods=["POST"])
def visualize_whole_data():
    graphic = data_manager.visualize_data(data_manager.X, data_manager.y, "Whole Data")
    if graphic:
        return render_template("visualization.html", graphic=graphic)
    else:
        flash("Error visualizing whole data.", "danger")
        return redirect(url_for("visualization"))


@app.route("/visualize_training", methods=["POST"])
def visualize_training_data():
    graphic = data_manager.visualize_data(
        data_manager.X_train, data_manager.y_train, "Training Data"
    )
    if graphic:
        return render_template("visualization.html", graphic=graphic)
    else:
        flash("Error visualizing training data.", "danger")
        return redirect(url_for("visualization"))


@app.route("/visualize_testing", methods=["POST"])
def visualize_testing_data():
    graphic = data_manager.visualize_data(
        data_manager.X_test, data_manager.y_test, "Testing Data"
    )
    if graphic:
        return render_template("visualization.html", graphic=graphic)
    else:
        flash("Error visualizing testing data.", "danger")
        return redirect(url_for("visualization"))


@app.route("/scaling")
def scaling():
    return render_template("scaling.html")


@app.route("/scale", methods=["POST"])
def scale_data():
    input_scaling_method = request.form["input_method"]
    target_scaling_method = request.form["target_method"]

    data_manager.scale_data(input_scaling_method, target_scaling_method)
    flash("Data scaled successfully!", "success")
    return redirect(url_for("scaling"))


@app.route("/train")
def training():
    return render_template("training.html")


@app.route("/train_model", methods=["POST"])
def train_model():
    model_name = request.form["model"]
    # Train the selected machine learning model and store the results
    # You can add your model training logic here and flash training results
    flash(f"Model trained successfully: {model_name}", "success")
    return redirect(url_for("training"))


@app.route("/validation")
def validation():
    return render_template("validation.html")


@app.route("/validate_model", methods=["POST"])
def validate_model():
    validation_metric = request.form["validation_metric"]
    # Validate the trained model using the selected validation metric and store the results
    # You can add your model validation logic here and flash validation results
    flash(f"Model validated using {validation_metric}", "success")
    return redirect(url_for("validation"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8085)
