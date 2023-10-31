import secrets
from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
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
                # Flush input columns on new file upload.
                session.pop("input_columns", None)
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
    data_manager.selected_input_column = request.form.get("input_column")
    data_manager.selected_target_column = request.form.get("target_column")
    # Check if the required keys are present in the form data
    if data_manager.selected_input_column is None or data_manager.selected_target_column is None:
        flash("Please select input and target columns!", "danger")
        return redirect(url_for("index"))
    try:
        data_manager.split_data(test_size)
    except Exception as e:
        flash(f"Error: {e}", "danger")
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
    graphic = data_manager.visualize_data(data_manager.x, data_manager.y, "Whole Data")
    if graphic:
        return render_template("visualization.html", graphic=graphic)
    else:
        flash("Error visualizing whole data.", "danger")
        return redirect(url_for("visualization"))

@app.route("/visualize_training", methods=["POST"])
def visualize_training_data():
    graphic = data_manager.visualize_data(
        data_manager.x_train, data_manager.y_train, "Training Data"
    )
    if graphic:
        return render_template("visualization.html", graphic=graphic)
    else:
        flash("Error visualizing training data.", "danger")
        return redirect(url_for("visualization"))

@app.route("/visualize_testing", methods=["POST"])
def visualize_testing_data():
    graphic = data_manager.visualize_data(
        data_manager.x_test, data_manager.y_test, "Testing Data"
    )
    if graphic:
        return render_template("visualization.html", graphic=graphic)
    else:
        flash("Error visualizing testing data.", "danger")
        return redirect(url_for("visualization"))

@app.route("/scaling")
def scaling():
    """Display the first five rows of the dataset."""

    first_five_data = data_manager.data.head().to_html() if data_manager.data is not None else None

    return render_template(
        "scaling.html",
        columns=data_manager.columns,
        first_five_data=first_five_data,
    )

@app.route("/scale", methods=["POST"])
def scale_data():
    scaling_method = request.form["scaling_method"]

    data_manager.scale_data(scaling_method)

    return render_template(
        "scaling.html",
        columns=data_manager.columns,
        first_five_x_scaled=pd.DataFrame(data_manager.x_train_scaled).head().to_html(),
        first_five_y_scaled=pd.DataFrame(data_manager.y_train_scaled).head().to_html(),
        input_column = data_manager.selected_input_column,
        target_column = data_manager.selected_target_column
    )

@app.route("/train")
def training():
    return render_template("training.html")

@app.route("/train_model", methods=["POST"])
def train_model():
    graphic = data_manager.model_training(pd.DataFrame(data_manager.x_train), pd.DataFrame(data_manager.y_train))

    if graphic:
        return render_template("training.html", graphic=graphic)
    else:
        flash("Error visualizing data for trained model.", "danger")
        return redirect(url_for("training"))

@app.route("/evaluation")
def evaluation():
    return render_template("evaluation.html")

@app.route("/evaluate_model", methods=["POST"])
def evaluate_model():
    evaluation_metric = request.form["evaluation_metric"]

    if evaluation_metric == "mean_squared_error":
        result = data_manager.calculate_rmse(pd.DataFrame(data_manager.y_test), pd.DataFrame(data_manager.y_pred))
        flash(f"MAE: {result[0]}", "success")
        flash(f"MSE: {result[1]}", "success")
        flash(f"RMSE: {result[2]}", "success")
    elif evaluation_metric == "accuracy_score":
        result = data_manager.calculate_accuracy(pd.DataFrame(data_manager.y_test), pd.DataFrame(data_manager.y_pred))
        flash(f"Accuracy Score: {result}", "success")
    else:
        flash("Invalid validation metric selected", "danger")

    return redirect(url_for("evaluation"))

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

