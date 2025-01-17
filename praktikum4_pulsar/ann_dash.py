import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import subprocess

# File paths for saved artifacts
MODEL_PATH = "dff_model.h5"
HISTORY_PATH = "dff_training_history.json"
X_TEST_PATH = "dff_X_test.npy"
Y_TEST_PATH = "dff_y_test.npy"

# Dash app
app = Dash(__name__)

# Helper functions for visualization
def create_learning_curves(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=history['accuracy'], mode='lines+markers', name='Train Accuracy'))
    fig.add_trace(go.Scatter(
        y=history['val_accuracy'], mode='lines+markers', name='Validation Accuracy'))
    fig.add_trace(go.Scatter(
        y=history['loss'], mode='lines+markers', name='Train Loss'))
    fig.add_trace(go.Scatter(
        y=history['val_loss'], mode='lines+markers', name='Validation Loss'))
    fig.update_layout(
        title="Learning Curves",
        xaxis_title="Epochs",
        yaxis_title="Metrics",
        legend_title="Metrics"
    )
    return fig

def create_confusion_matrix(conf_matrix):
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=["Predicted 0", "Predicted 1"],
        y=["Actual 0", "Actual 1"],
        colorscale="Blues",
        showscale=True,
        text=conf_matrix,
        texttemplate="%{text}"
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )
    return fig

# App layout
app.layout = html.Div([
    html.H1("Deep Feedforward Neural Network Visualization"),
    
    html.Div([
        html.Label("Choose an action:"),
        dcc.RadioItems(
            id='train-new-model',
            options=[
                {'label': 'Use existing model', 'value': 'existing'},
                {'label': 'Train a new model', 'value': 'new'}
            ],
            value='existing',  # Default is to use the existing model
            inline=True
        ),
    ]),
    
    html.Div(id='status', style={'marginTop': 20}),
    
    # Learning Curves
    html.H2("Learning Curves"),
    dcc.Graph(id="learning-curves"),
    
    # Confusion Matrix
    html.H2("Confusion Matrix"),
    dcc.Graph(id="confusion-matrix"),
])

@app.callback(
    [Output("status", "children"),
    Output("learning-curves", "figure"),
    Output("confusion-matrix", "figure")],
    [Input("train-new-model", "value")]
)
def update_graphs(action):
    if action == "new":
        # Train a new model by running the training script
        subprocess.run(["python3", "dff_model.py"])
    
    # Load the model and data
    model = load_model(MODEL_PATH)
    with open(HISTORY_PATH, 'r') as f:
        history = json.load(f)
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    
    # Generate predictions and confusion matrix
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Create status message
    status_message = "Trained a new model." if action == "new" else "Using the existing model."
    
    # Generate visualizations
    learning_curves = create_learning_curves(history)
    confusion_matrix_fig = create_confusion_matrix(conf_matrix)
    
    return status_message, learning_curves, confusion_matrix_fig

if __name__ == "__main__":
    app.run_server(debug=True, port=8060)