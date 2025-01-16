import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import plot_model

from dash import Dash, dcc, html, Input as DashInput, Output
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

import matplotlib.pyplot as plt

# 1. Load and prepare the dataset
# get to know the dataset
data = pd.read_csv("pulsar_data.csv")
data_head = data.head()
data_tail = data.tail()
# data_info = data.info()

# handle missing values by replacing them with the mean of the column
data_cleaned = data.fillna(data.mean())

# feature X and target y
X = data_cleaned.drop(columns = ["target_class"])
y = data_cleaned["target_class"]

# print(X.info())

# split the data into training and test data (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# standardize the data (do I need it?)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PCA: Principal Component Analysis



# 2. SVM
# parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'gamma': [0.01, 0.1, 1],
    'degree': [2, 3, 4]
}

def train_clf (C, gamma, degree):
    models = (
        SVC(kernel='linear', C=C),
        SVC(kernel='poly', C=C, degree=degree),
        SVC(kernel='rbf', C=C, gamma=gamma),
        SVC(kernel='sigmoid', C=C, gamma=gamma)
    )
    clfs = (model.fit(X_train, y_train) for model in models)

    return clfs

# def evaluate_clf():

# def get_best_clf():




app = Dash(__name__)

# Define the scatter plot using Plotly
fig = go.Figure()

# Add points for each class
for class_value in np.unique(y):
    fig.add_trace(go.Scatter(
        x=X[y == class_value, 0],
        y=X[y == class_value, 1],
        mode='markers',
        marker=dict(size=10, line=dict(width=2), symbol='circle'),
        name=f'Class {class_value}'
    ))

# Update the layout
fig.update_layout(
    title="Samples in two-dimensional feature space",
    xaxis=dict(title="Feature 1", range=[-3, 3]),
    yaxis=dict(title="Feature 2", range=[-3, 3]),
    legend=dict(title="Classes"),
    width=500,
    height=400
)

# Define the Dash layout
app.layout = html.Div([
    html.H1("2D Feature Space Visualization"),
    dcc.Graph(figure=fig)
])


if __name__ == "__main__":
    app.run_server(debug=True)