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
from module.input import DataModelManager

matplotlib.use("Agg")

app = Flask(__name__)
app.secret_key = secrets.token_hex(24)

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

@app.route("/clean", methods=["POST"])
def clean_data():
    data_manager.remove_NaN_values()
    return redirect(url_for("index"))

@app.route("/remove_duplicates", methods=["POST"])
def remove_duplicates():
    data_manager.remove_duplicates()
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
    # If no file is uploaded, redirect to the home page.
    if data_manager.data is None:
        flash("Please upload a file first!", "danger")
        return redirect(url_for("index"))
    else:
        return render_template("visualization.html", columns=data_manager.columns)

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
    """Display the first five rows of the dataset."""

    if data_manager.data is not None:
        return render_template(
            "scaling.html",
            columns=data_manager.columns,
            first_five=data_manager.data.head().to_html(),
        )
    else:
        flash("Please upload a CSV file.", "danger")
        return redirect(url_for("index"))

@app.route("/scale", methods=["POST"])
def scale_data():
    input_scaling_method = request.form["input_method"]
    target_scaling_method = request.form["target_method"]

    data_manager.scale_data(input_scaling_method, target_scaling_method)
    flash("Data scaled successfully!", "success")

    first_5_columns_X_scaled = data_manager.X_scaled[:, :5]
    first_5_columns_y_scaled = data_manager.y_scaled[:, :5]
    return render_template(
        "scaled_data.html",
        columns=data_manager.columns,
        first_5_columns_X_scaled=first_5_columns_X_scaled,
        first_5_columns_y_scaled=first_5_columns_y_scaled,
    )

@app.route("/train")
def training():
    return render_template("training.html")

@app.route("/train_model", methods=["POST"])
def train_model():
    model_name = request.form["model"]

    flash(f"Model trained successfully: {model_name}", "success")
    return redirect(url_for("training"))

@app.route("/evaluation")
def evaluation():
    return render_template("evaluation.html")

@app.route("/evaluate_model", methods=["POST"])
def evaluate_model():
    validation_metric = request.form["validation_metric"]
    # Validate the trained model using the selected validation metric and store the results
    # You can add your model validation logic here and flash validation results
    flash(f"Model validated using {validation_metric}", "success")
    return redirect(url_for("validation"))

@app.route("/data")
def show_data_table():
    """Show to the CSV file as a table."""

    if data_manager.data is not None:
        return render_template("data.html", data_table=data_manager.data.to_html())
    else:
        flash("Please upload a CSV file.", "danger")
        return redirect(url_for("index"))

@app.route("/export")
def export():
    return render_template("export.html")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8085, debug=True)
