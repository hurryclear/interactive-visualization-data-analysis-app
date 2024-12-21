# Import libraries
import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import plotly.express as px
from dash import Dash, dcc, html
import matplotlib.pyplot as plt
import io
import base64

# 1. DATA PREPARATION
df = pd.read_csv('titanic.csv')

# Clean the data
# These columns (PassengerId, Name, Ticket, Cabin) are not useful for predicting survival. They either contain unique identifiers or non-informative text data that do not contribute to the model's predictive power.
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], errors='ignore')
# The 'Age' column has missing values. Using the median to fill these gaps is a common practice because it is robust to outliers and provides a central tendency measure.
df['Age'] = np.floor(df['Age'])
df['Age'].fillna(df['Age'].median(), inplace=True)
# The 'Embarked' column has missing values. Filling them with the most frequent value.
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
# Features and target: To prepare the data for modeling, we need to separate the features (input variables) from the target (output variable).
X = df.drop(columns=['Survived'])
y = df['Survived']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3)
}

# train the model
def train_model(model, data_train):
    """Trains the model on the training data."""
    for data in data_train:
        X_train, y_train = data[:2]  # Only use the first two values
        model.fit(X_train, y_train)
    return model

# cross-validation splits data
def cross_validation_data(X, y):
    """Generates train-test splits for cross-validation."""
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    data_train = []
    data_test = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        data_train.append((X_train, y_train))
        data_test.append((X_test, y_test))

    return data_train, data_test


# Cross-validation method
def cross_validate_evaluation(trained_model, data_test):
    """Perform cross-validation and return metrics and confusion matrix."""
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} # Store metrics for each fold
    confusion_matrices = [] # Store confusion matrices for each fold

    for X_test, y_test in data_test:
        # predict the data
        y_pred = trained_model.predict(X_test)
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        metrics['recall'].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        metrics['f1'].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        confusion_matrices.append(confusion_matrix(y_test, y_pred))

    avg_confusion_matrix = np.mean(confusion_matrices, axis=0)
    avg_metrics = {key: np.mean(value) for key, value in metrics.items()}
    
    return { # return a dictionary containing the average metrics and the average confusion matrix.
        **avg_metrics, # Unpack metrics
        'confusion_matrix': avg_confusion_matrix
        # 'decision_tree': dt_image
    }

# bootstrap splits data
def bootstrap_632_data(X, y):
    """Generates train-test splits for the Bootstrap .632 method."""
    data = []
    for _ in range(100):  # Number of bootstrap iterations
        X_boot, y_boot = resample(X, y, replace=True, n_samples=len(X), random_state=42)
        
        oob_indices = np.setdiff1d(np.arange(len(X)), np.unique(X_boot, return_index=True)[1])
        X_oob, y_oob = X[oob_indices], y[oob_indices]
        data.append((X_boot, y_boot, X_oob, y_oob, oob_indices))

    return data

# bootstrap evaluation
def bootstrap_632_evaluation(trained_model, data):
    train_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    test_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    confusion_matrices = []

    for X_boot, y_boot, X_oob, y_oob, oob_indices in data:

        y_pred_train = trained_model.predict(X_boot) 

        # Collect train metrics
        train_metrics['accuracy'].append(accuracy_score(y_boot, y_pred_train))
        train_metrics['precision'].append(precision_score(y_boot, y_pred_train, average='weighted', zero_division=0))
        train_metrics['recall'].append(recall_score(y_boot, y_pred_train, average='weighted', zero_division=0))
        train_metrics['f1'].append(f1_score(y_boot, y_pred_train, average='weighted', zero_division=0))

        # Collect test (OOB) metrics if there are out-of-bag samples
        if len(oob_indices) > 0:
            y_pred_oob = trained_model.predict(X_oob)
            test_metrics['accuracy'].append(accuracy_score(y_oob, y_pred_oob))
            test_metrics['precision'].append(precision_score(y_oob, y_pred_oob, average='weighted', zero_division=0))
            test_metrics['recall'].append(recall_score(y_oob, y_pred_oob, average='weighted', zero_division=0))
            test_metrics['f1'].append(f1_score(y_oob, y_pred_oob, average='weighted', zero_division=0))
            confusion_matrices.append(confusion_matrix(y_oob, y_pred_oob))

    # Aggregate train and test metrics
    train_metrics_mean = {metric: np.mean(train_metrics[metric]) for metric in train_metrics}
    test_metrics_mean = {metric: np.mean(test_metrics[metric]) for metric in test_metrics}

    # Combine metrics using the .632 formula
    combined_metrics = {metric: 0.368 * train_metrics_mean[metric] + 0.632 * test_metrics_mean[metric] for metric in train_metrics}
    avg_confusion_matrix = np.mean(confusion_matrices, axis=0) if confusion_matrices else None

    combined_metrics['confusion_matrix'] = avg_confusion_matrix
    return combined_metrics


# decision tree visualization for prediction
def decision_tree_visualization(trained_model, X):
    """
    """
    buf = io.BytesIO()
    plt.figure(figsize=(15, 10))  # Reduced figure size for better app performance
    plot_tree(trained_model, feature_names=X.columns, class_names=["Not Survived", "Survived"], filled=True)
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)  # Reset buffer position
    return base64.b64encode(buf.getvalue()).decode("utf8")

def visualize_decision_path(trained_model, feature_names, class_names, sample_index, X_test, y_test):
    """
    Visualize the decision path of a trained Decision Tree for a specific test sample.
    """

    # Extract the decision path for the test sample
    node_indicator = trained_model.decision_path(X_test)
    feature = trained_model.tree_.feature
    threshold = trained_model.tree_.threshold

    # Select the sample's path
    sample_path = node_indicator.indices[
        node_indicator.indptr[sample_index]:node_indicator.indptr[sample_index + 1]
    ]

    # Print decision path for the sample
    print(f"Decision path for sample {sample_index}:")
    for node_id in sample_path:
        if feature[node_id] != -2:  # Not a leaf node
            print(f"Node {node_id}: If {feature_names[feature[node_id]]} <= {threshold[node_id]}")
        else:
            print(f"Node {node_id}: Leaf node")

    # Visualize the decision tree with the decision path highlighted
    buf = io.BytesIO()
    plt.figure(figsize=(15, 10))
    plot_tree(trained_model, feature_names=feature_names, class_names=class_names, filled=True)
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf8")



# Evaluate models
crossval_results = {}
bootstrap_results = {}
decision_tree_cv = None
decision_tree_bs = None

# Convert X_scaled to DataFrame for compatibility
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# train the model, evaluate the model and store the results in the iteration
for model_name, model in models.items():
    # Cross-validation
    data_train_cv, data_test_cv = cross_validation_data(X_scaled, y)
    trained_model_cv = train_model(model, data_train_cv)
    crossval_results[model_name] = cross_validate_evaluation(trained_model_cv, data_test_cv)
    
    # Bootstrap
    data_bs = bootstrap_632_data(X_scaled, y)
    trained_model_bs = train_model(model, data_bs)
    bootstrap_results[model_name] = bootstrap_632_evaluation(trained_model_bs, data_bs)

    # Store decision tree visualization if the model is DecisionTreeClassifier
    if isinstance(model, DecisionTreeClassifier):
        decision_tree_cv = visualize_decision_path(trained_model_cv, X.columns, ["Not Survived", "Survived"], 0, X_scaled, y)
        decision_tree_bs = visualize_decision_path(trained_model_bs, X.columns, ["Not Survived", "Survived"], 0, X_scaled, y)


# 3. Build Dash App
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Titanic Classification Models Evaluation", style={'textAlign': 'center'}),

    # Metrics Comparison
    html.H3("Cross-Validation Metrics Comparison"),
    dcc.Graph(figure=px.bar(
        pd.DataFrame([
            {'Metric': metric, 'Model': model, 'Score': crossval_results[model][metric]}
            for model in models.keys()
            for metric in ['accuracy', 'precision', 'recall', 'f1']
        ]),
        x='Metric', y='Score', color='Model', barmode='group',
        title="Cross-Validation Metrics Comparison",
        labels={'Metric': 'Metrics', 'Score': 'Score'}
    )),

    html.H3("Bootstrap Metrics Comparison"),
    dcc.Graph(figure=px.bar(
        pd.DataFrame([
            {'Metric': metric, 'Model': model, 'Score': bootstrap_results[model][metric]}
            for model in models.keys()
            for metric in ['accuracy', 'precision', 'recall', 'f1']
        ]),
        x='Metric', y='Score', color='Model', barmode='group',
        title="Bootstrap Metrics Comparison",
        labels={'Metric': 'Metrics', 'Score': 'Score'}
    )),

    # Confusion Matrices
    html.H3("Confusion Matrices: Cross-Validation"),
    html.Div([
        html.Div([
            dcc.Graph(figure=px.imshow(crossval_results[model]['confusion_matrix'], text_auto=True,
                                        title=f"{model}",
                                        color_continuous_scale='Blues').update_layout(
                                            autosize=False,
                                            width=300,
                                            height=300
                                        ))
        ], style={'margin': '10px'})
        for model in models.keys()
    ], style={'display': 'flex', 'flexDirection': 'row', 'flexWrap': 'wrap'}),
    
    html.H3("Confusion Matrices: Bootstrap"),
    html.Div([
        html.Div([
            dcc.Graph(figure=px.imshow(bootstrap_results[model]['confusion_matrix'], text_auto=True,
                                        title=f"{model}",
                                        color_continuous_scale='Reds').update_layout(
                                            autosize=False,
                                            width=300,
                                            height=300
                                        ))
        ], style={'margin': '10px'})
        for model in models.keys()
    ], style={'display': 'flex', 'flexDirection': 'row', 'flexWrap': 'wrap'}),

    # Decision Tree visualisierung
    html.H3("Decision Tree(Crossvalidation)"),
    html.Img(src="data:image/png;base64,{}".format(decision_tree_cv), style={'width': '100%', 'height': 'auto'}),
    html.H3("Decision Tree(Bootstrap)"),
    html.Img(src="data:image/png;base64,{}".format(decision_tree_bs), style={'width': '100%', 'height': 'auto'}),

])

if __name__ == '__main__':
    app.run_server(debug=True)