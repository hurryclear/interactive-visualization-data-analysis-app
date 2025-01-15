# Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Load Dataset
data_path = "pulsar_data.csv"  # Ensure this file is in the same directory as the script
data = pd.read_csv(data_path)

# Data Preprocessing
data = data.fillna(data.mean())  # Handle missing values
X = data.drop(columns=["target_class"])
y = data["target_class"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM Parameter Grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf']
}

# Perform Grid Search
grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='accuracy', verbose=0)
grid_search.fit(X_train, y_train)

# Results from Grid Search
results = pd.DataFrame(grid_search.cv_results_)

# Best Model
best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_svm = grid_search.best_estimator_

# Evaluate Best Model on Test Data
y_pred = best_svm.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
test_report = classification_report(y_test, y_pred)

# Dash App Setup
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("SVM Classification and Parameter Optimization"),
    
    html.Div([
        html.H2("Best Parameters"),
        html.P(f"C: {best_params['C']}"),
        html.P(f"Gamma: {best_params['gamma']}"),
        html.P(f"Kernel: {best_params['kernel']}"),
        html.P(f"Cross-Validation Accuracy: {best_score:.2f}"),
        html.P(f"Test Accuracy: {test_accuracy:.2f}")
    ]),

    html.Div([
        html.H2("Classification Report"),
        html.Pre(test_report)
    ]),

    html.Div([
        html.H2("Parameter Tuning Results"),
        dcc.Graph(
            id='svm-parameter-visualization',
            figure={
                'data': [
                    go.Scatter3d(
                        x=results['param_C'].astype(float),
                        y=results['param_gamma'].astype(float),
                        z=results['mean_test_score'],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=results['mean_test_score'],
                            colorscale='Viridis',
                            opacity=0.8
                        ),
                        text=[f"Kernel: {k}" for k in results['param_kernel']]
                    )
                ],
                'layout': go.Layout(
                    title="SVM Parameter Tuning Results",
                    scene={
                        'xaxis_title': 'C (Regularization)',
                        'yaxis_title': 'Gamma',
                        'zaxis_title': 'Mean CV Accuracy'
                    }
                )
            }
        )
    ])
])

# Run the App
if __name__ == "__main__":
    app.run_server(debug=True)