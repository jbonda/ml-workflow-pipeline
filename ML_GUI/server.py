import os
import secrets
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
from io import BytesIO
import random
import base64
import matplotlib
matplotlib.use('Agg')

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

    def load_data(self, file):
        try:
            if file.filename.endswith('.csv'):
                self.data = pd.read_csv(file)
                self.columns = list(self.data.columns)  # Store the column names
                return True
            else:
                flash('Please upload a CSV file.', 'danger')
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
        return False

    def split_data(self, test_size):
        if self.data is not None and self.selected_input_column and self.selected_target_column:
            self.X = self.data[[self.selected_input_column]]
            self.y = self.data[self.selected_target_column]
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
            self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test  # Store training and testing data
            flash('Data split successfully!', 'success')
        else:
            flash('Please select input and target columns and upload data.', 'danger')

    def visualize_data(self, X, y, title):
        if X is not None and y is not None:
            fig, axes = plt.subplots(figsize=(8, 6))
            axes.scatter(X, y)
            plt.show()
            axes.set_title(f'Scatter Plot - {title}')
            axes.set_xlabel(self.selected_input_column)
            axes.set_ylabel(self.selected_target_column)
            plt.tight_layout()

            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()

            graphic = base64.b64encode(image_png).decode()
            return graphic
        else:
            flash('Invalid data for visualization.', 'danger')
            return None



data_manager = DataModelManager()

@app.route('/')
def index():
    if 'tab_id' not in session:
        session['tab_id'] = secrets.token_hex(24)
        session.permanent = False
    return render_template('index.html', columns=data_manager.columns)

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            if data_manager.load_data(file):
                flash('File uploaded successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/split', methods=['POST'])
def split_data():
    test_size = float(request.form['test_size'])
    data_manager.selected_input_column = request.form['input_column']
    data_manager.selected_target_column = request.form['target_column']
    data_manager.split_data(test_size)
    return redirect(url_for('index'))

@app.route('/visualize_whole', methods=['POST'])
def visualize_whole_data():
    if data_manager.X is not None and data_manager.y is not None:
        graphic = data_manager.visualize_data(data_manager.X, data_manager.y, 'Whole data')
        if graphic:
            return render_template('index.html', graphic=graphic, columns=data_manager.columns)
    flash('Upload a data file!.', 'danger')
    return redirect(url_for('index'))

@app.route('/visualize_training', methods=['POST'])
def visualize_training_data():
    if data_manager.X_train is not None and data_manager.y_train is not None:
        graphic = data_manager.visualize_data(data_manager.X_train, data_manager.y_train, 'Training Data')
        if graphic:
            return render_template('index.html', graphic=graphic, columns=data_manager.columns)
    flash('No training data available for visualization.', 'danger')
    return redirect(url_for('index'))

@app.route('/visualize_testing', methods=['POST'])
def visualize_testing_data():
    if data_manager.X_test is not None and data_manager.y_test is not None:
        graphic = data_manager.visualize_data(data_manager.X_test, data_manager.y_test, 'Testing Data')
        if graphic:
            return render_template('index.html', graphic=graphic, columns=data_manager.columns)
    flash('No testing data available for visualization.', 'danger')
    return redirect(url_for('index'))

@app.route('/visualize')
def viz():
    pass

@app.route('/model')
def hyperparameters():
    return render_template('model.html')

@app.route('/export')
def conclusion():
    return send_from_directory("client/public", "export.html")

# Path for all the static files (compiled JS/CSS, etc.)
# @app.route("/<path:path>")
# def base(path):
#     return send_from_directory("client/public", path)

if __name__ == '__main__':
    app.run(debug=True)

