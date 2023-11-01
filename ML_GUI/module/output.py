# Output Module

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def calculate_rmse(y_true, y_pred):
    """Calculate generated evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, mse, rmse

def calculate_accuracy(y_true, y_pred, threshold=0.5):
    """Calculate accuracy score."""
    y_pred_binary = (y_pred > threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred_binary)
    return accuracy

def design_format():
    """Plot formatting and conversion to base64 encoding."""
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    graphic = base64.b64encode(image_png).decode()
    return graphic

def export_model():
    """Method to export the model."""
    return None

def download_code():
    """Method to download and verify the corresponding system code."""
    return None
