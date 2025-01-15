import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

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
pulsar_data = pd.read_csv("pulsar_data.csv")
pulsar_data_head = pulsar_data.head()
pulsar_data_tail = pulsar_data.tail()
# pulsar_data_info = pulsar_data.info()

# handle missing values by replacing them with the mean of the column
pulsar_data_cleaned = pulsar_data.fillna(pulsar_data.mean())

# feature X and target y
X = pulsar_data_cleaned.drop(columns = ["target_class"])
y = pulsar_data_cleaned["target_class"]

# print(X.info())

# split the data into training and test data (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 2. SVM
# initialize the svm classifier with default parameters
svm_classifier = SVC(random_state=42)
# parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.1, 0.01],
    'kernel': ['linear', 'rbf']
}
# train the classifier with the training data
grid_search = GridSearchCV(svm_classifier, param_grid, refit=True, verbose=2)
grid_search.fit(X_train, y_train)

# predict the target values of the test data
y_pred = grid_search.predict(X_test)

# evaluate the classifier
svm_accuracy = accuracy_score(y_test, y_pred)
svm_classification_report = classification_report(y_test, y_pred, output_dict=True)
svm_precision = svm_classification_report['weighted avg']['precision']
svm_recall = svm_classification_report['weighted avg']['recall']
svm_f1 = svm_classification_report['weighted avg']['f1-score']

# Plot learning curves
def plot_learning_curve(history, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(history.history['accuracy']))), y=history.history['accuracy'], mode='lines', name='Train Accuracy'))
    fig.add_trace(go.Scatter(x=list(range(len(history.history['val_accuracy']))), y=history.history['val_accuracy'], mode='lines', name='Validation Accuracy'))
    fig.update_layout(title=f'{title} - Accuracy', xaxis_title='Epochs', yaxis_title='Accuracy')

    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=list(range(len(history.history['loss']))), y=history.history['loss'], mode='lines', name='Train Loss'))
    fig_loss.add_trace(go.Scatter(x=list(range(len(history.history['val_loss']))), y=history.history['val_loss'], mode='lines', name='Validation Loss'))
    fig_loss.update_layout(title=f'{title} - Loss', xaxis_title='Epochs', yaxis_title='Loss')

    return fig, fig_loss

# 3. ANN
# ANN model1
model1 = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Train ANN model1
history1 = model1.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test), verbose=0)
ann1_accuracy = model1.evaluate(X_test, y_test, verbose=0)[1]
ann1_learning_curve, ann1_loss_curve = plot_learning_curve(history1, 'ANN1')

# ANN model2
model2 = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train ANN model2
history2 = model2.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test), verbose=0)
ann2_accuracy = model2.evaluate(X_test, y_test, verbose=0)[1]
ann2_learning_curve, ann2_loss_curve = plot_learning_curve(history2, 'ANN2')

# Evaluate the models
ann1_accuracy = model1.evaluate(X_test, y_test, verbose=0)[1]
ann2_accuracy = model2.evaluate(X_test, y_test, verbose=0)[1]

# Calculate precision, recall, and f1-score for ANN models
ann1_pred = (model1.predict(X_test) > 0.5).astype("int32")
ann2_pred = (model2.predict(X_test) > 0.5).astype("int32")

ann1_precision = precision_score(y_test, ann1_pred, average='weighted')
ann1_recall = recall_score(y_test, ann1_pred, average='weighted')
ann1_f1 = f1_score(y_test, ann1_pred, average='weighted')

ann2_precision = precision_score(y_test, ann2_pred, average='weighted')
ann2_recall = recall_score(y_test, ann2_pred, average='weighted')
ann2_f1 = f1_score(y_test, ann2_pred, average='weighted')

# 4. Dash App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H1("Pulsar Data Classification", style={'textAlign': 'center'}),
    
    # Dropdown for selecting the model
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'SVM', 'value': 'svm'},
            {'label': 'ANN1', 'value': 'ann1'},
            {'label': 'ANN2', 'value': 'ann2'}
        ],
        value='svm'
    ),
    
    # Graph for visualizing the results
    dcc.Graph(id='results-graph'),
    
    # Graph for visualizing the network topology
    dcc.Graph(id='topology-graph'),

    # Graph for visualizing the learning curve
    dcc.Graph(id='learning-curve-graph'),

    # Graph for visualizing the loss curve
    dcc.Graph(id='loss-curve-graph')
])

@app.callback(
    [Output('results-graph', 'figure'),
     Output('topology-graph', 'figure'),
     Output('learning-curve-graph', 'figure'),
     Output('loss-curve-graph', 'figure')],
    [DashInput('model-dropdown', 'value')]
)
def update_graph(selected_model):
    if selected_model == 'svm':
        fig = go.Figure(data=[
            go.Bar(name='Accuracy', x=['SVM'], y=[svm_accuracy]),
            go.Bar(name='Precision', x=['SVM'], y=[svm_precision]),
            go.Bar(name='Recall', x=['SVM'], y=[svm_recall]),
            go.Bar(name='F1-Score', x=['SVM'], y=[svm_f1])
        ])
        fig.update_layout(barmode='group', title='SVM Performance Metrics')
        topology_fig = go.Figure()
        learning_curve_fig = go.Figure()
        loss_curve_fig = go.Figure()
    elif selected_model == 'ann1':
        fig = go.Figure(data=[
            go.Bar(name='Accuracy', x=['ANN1'], y=[ann1_accuracy]),
            go.Bar(name='Precision', x=['ANN1'], y=[ann1_precision]),
            go.Bar(name='Recall', x=['ANN1'], y=[ann1_recall]),
            go.Bar(name='F1-Score', x=['ANN1'], y=[ann1_f1])
        ])
        fig.update_layout(barmode='group', title='ANN1 Performance Metrics')
        topology_fig = go.Figure()
        plot_model(model1, to_file='model1.png', show_shapes=True, show_layer_names=True)
        topology_fig.add_layout_image(dict(source='model1.png', xref="x", yref="y", x=0, y=1, sizex=1, sizey=1, xanchor="left", yanchor="top"))
        learning_curve_fig = ann1_learning_curve
        loss_curve_fig = ann1_loss_curve
    elif selected_model == 'ann2':
        fig = go.Figure(data=[
            go.Bar(name='Accuracy', x=['ANN2'], y=[ann2_accuracy]),
            go.Bar(name='Precision', x=['ANN2'], y=[ann2_precision]),
            go.Bar(name='Recall', x=['ANN2'], y=[ann2_recall]),
            go.Bar(name='F1-Score', x=['ANN2'], y=[ann2_f1])
        ])
        fig.update_layout(barmode='group', title='ANN2 Performance Metrics')
        topology_fig = go.Figure()
        plot_model(model2, to_file='model2.png', show_shapes=True, show_layer_names=True)
        topology_fig.add_layout_image(dict(source='model2.png', xref="x", yref="y", x=0, y=1, sizex=1, sizey=1, xanchor="left", yanchor="top"))
        learning_curve_fig = ann2_learning_curve
        loss_curve_fig = ann2_loss_curve
    
    return fig, topology_fig, learning_curve_fig, loss_curve_fig

if __name__ == '__main__':
    app.run_server(debug=True)