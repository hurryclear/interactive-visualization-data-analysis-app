import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from dash import Dash, dcc, html, Input, Output, dash_table
from dff_model import create_learning_curves, create_confusion_matrix

# File paths for saved artifacts
MODEL_PATH = "dff_model.h5"
HISTORY_PATH = "dff_training_history.json"
EVAL_PATH = "dff_evaluation_metrics.json"

# Dash app
app = Dash(__name__)

# Helper functions for visualization

# App layout
app.layout = html.Div([
    html.H1("Deep Feedforward Neural Network Visualization"),
    
    # Evaluation Metrics
    html.H2("Evaluation Metrics"),
    dash_table.DataTable(
        id="evaluation-metrics",
        style_table={"margin": "20px auto", "width": "60%"},
        style_header={"backgroundColor": "lightgrey", "fontWeight": "bold"},
        style_cell={"textAlign": "center", "padding": "10px"},
    ),

    # Learning Curves
    html.H2("Learning Curves"),
    dcc.Graph(id="learning-curves"),
    
    # Confusion Matrix
    html.H2("Confusion Matrix"),
    dcc.Graph(id="confusion-matrix"),
])

@app.callback(
    [Output("learning-curves", "figure"),
    Output("confusion-matrix", "figure")],
    [Input("learning-curves", "id")]  # A dummy input to trigger the callback once
)
def update_graphs(_):
    # Load training history
    with open(HISTORY_PATH, 'r') as f:
        history = json.load(f)

    # Load evaluation metrics
    with open(EVAL_PATH, 'r') as f:
        evaluation = json.load(f)
    conf_matrix = np.array(evaluation["confusion_matrix"])

    # Generate visualizations
    learning_curves = create_learning_curves(history)
    confusion_matrix_fig = create_confusion_matrix(conf_matrix)
    
    return learning_curves, confusion_matrix_fig

if __name__ == "__main__":
    app.run_server(debug=True, port=8060)