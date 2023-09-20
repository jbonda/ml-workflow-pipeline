import os
import secrets
from flask import Flask, render_template, request, redirect, url_for, flash, session
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
        self.lr_model = None
        self.logistic_model = None

    def load_data(self, file):
        try:
            if file.filename.endswith('.csv'):
                self.data = pd.read_csv(file)
                self.X = self.data[['smoking']]
                self.y = self.data['heart.disease']
                return True
            else:
                flash('Please upload a CSV file.', 'danger')
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
        return False

    def split_data(self, test_size):
        if self.X is not None and self.y is not None:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
            flash('Data split successfully!', 'success')
        else:
            flash('Please upload a data file first!', 'danger')
    
    def visualize_data(self):
        if self.data is not None:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

            # Scatter plot
            axes[0].scatter(self.data['smoking'], self.data['heart.disease'])
            axes[0].set_title('Scatter Plot')
            axes[0].set_xlabel('smoking')
            axes[0].set_ylabel('heart.disease')

            # Bar plot
            axes[1].bar(self.data['smoking'], self.data['heart.disease'])
            axes[1].set_title('Bar Plot')
            axes[1].set_xlabel('smoking')
            axes[1].set_ylabel('heart.disease')


            plt.tight_layout()

            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()

            graphic = base64.b64encode(image_png).decode()
            return graphic
        else:
            flash('Please upload a data file first!', 'danger')
            return None        

data_manager = DataModelManager()

@app.route('/')
def index():
    # Create a unique session ID for each tab and make it non-permanent
    if 'tab_id' not in session:
        session['tab_id'] = secrets.token_hex(24)
        session.permanent = False
    return render_template('index.html')

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
    data_manager.split_data(test_size)
    return redirect(url_for('index'))

@app.route('/visualize', methods=['GET', 'POST'])
def visualize_data():
    graphic = data_manager.visualize_data()
    if graphic:
        return render_template('index.html', graphic=graphic)
    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)
