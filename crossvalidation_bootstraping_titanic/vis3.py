import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier, plot_tree
import plotly.express as px
from dash import Dash, dcc, html
import io
import base64
import matplotlib.pyplot as plt


# 1. DATA PREPARATION
df = pd.read_csv('titanic.csv')

# Clean the data
# These columns (PassengerId, Name, Ticket, Cabin) are not useful for predicting survival. They either contain unique identifiers or non-informative text data that do not contribute to the model's predictive power.
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], errors='ignore')
# The 'Age' column has missing values. Using the median to fill these gaps is a common practice because it is robust to outliers and provides a central tendency measure.
df['Age'] = np.floor(df['Age'])
df['Age'].fillna(df['Age'].median(), inplace=True)

# The 'Embarked' column has missing values. Filling them with the most frequent value.
# df['Embarked'] = df['Embarked'].astype('category')
# df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# df['Sex'] = df['Sex'].astype('category')

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)


# Features and target: To prepare the data for modeling, we need to separate the features (input variables) from the target (output variable).
X = df.drop(columns=['Survived'])
y = df['Survived']

# Feature Scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# Models array (2D array for Cross-Validation and Bootstrap)
models = {
    "Logistic Regression": [LogisticRegression(max_iter=1000), LogisticRegression(max_iter=1000)],
    "Decision Tree": [DecisionTreeClassifier(random_state=42), DecisionTreeClassifier(random_state=42)],
    "KNN (k=3)": [KNeighborsClassifier(n_neighbors=3), KNeighborsClassifier(n_neighbors=3)],
}

# def train_model(model, data_train):
#     """Trains the model on the training data."""
#     for data in data_train:
#         X_train, y_train = data[:2]  # Only use the first two values
#         model.fit(X_train, y_train)
#     return model

# def cross_validation_data(X, y):
#     """Generates train-test splits for cross-validation."""
#     cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

#     data_train = []
#     data_test = []

#     for train_idx, test_idx in cv.split(X, y):
#         X_train, X_test = X[train_idx], X[test_idx]
#         y_train, y_test = y[train_idx], y[test_idx]
#         data_train.append((X_train, y_train))
#         data_test.append((X_test, y_test))

#     return data_train, data_test

# Cross-validation method
def cross_validate_method(model, X, y):
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    confusion_matrices = []
    # all_predictions = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx] if isinstance(X, pd.DataFrame) else (X[train_idx], X[test_idx])
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx] if isinstance(y, pd.Series) else (y[train_idx], y[test_idx])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # all_predictions.append((X_test, y_test, y_pred))  # Store predictions and test labels

        metrics['accuracy'].append(accuracy_score(y[test_idx], y_pred))
        metrics['precision'].append(precision_score(y[test_idx], y_pred, average='weighted', zero_division=0))
        metrics['recall'].append(recall_score(y[test_idx], y_pred, average='weighted', zero_division=0))
        metrics['f1'].append(f1_score(y[test_idx], y_pred, average='weighted', zero_division=0))
        confusion_matrices.append(confusion_matrix(y[test_idx], y_pred))

    avg_confusion_matrix = np.mean(confusion_matrices, axis=0)
    avg_metrics = {key: np.mean(value) for key, value in metrics.items()}

    return { 
        **avg_metrics,
        'confusion_matrix': avg_confusion_matrix
        # 'decision_tree': dt_image
    }

# Bootstrap method
def bootstrap_632_method(model, X, y):
    train_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    test_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    confusion_matrices = []
    # all_predictions = []

    for _ in range(100):  
        # Resample for bootstrap
        X_train, y_train = resample(X, y, replace=True, n_samples=len(X), random_state=42)

        # get incides for out-of-bag samples (for test)
        test_indices = np.setdiff1d(np.arange(len(X)), np.unique(X_train, return_index=True)[1])

        # Verwende .iloc[] für pandas DataFrame/Series 
        X_test = X.iloc[test_indices] if isinstance(X, pd.DataFrame) else X[test_indices]
        y_test = y.iloc[test_indices] if isinstance(y, pd.Series) else y[test_indices]

        # Trainiere das Modell mit den Bootstrap-Daten
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)

        train_metrics['accuracy'].append(accuracy_score(y_train, y_pred_train))
        train_metrics['precision'].append(precision_score(y_train, y_pred_train, average='weighted', zero_division=0))
        train_metrics['recall'].append(recall_score(y_train, y_pred_train, average='weighted', zero_division=0))
        train_metrics['f1'].append(f1_score(y_train, y_pred_train, average='weighted', zero_division=0))

        if len(test_indices) > 0:
            
            y_pred_oob = model.predict(X_test)
            # all_predictions.append((X_test, y_test, y_pred_oob))  # Store predictions and test labels

            test_metrics['accuracy'].append(accuracy_score(y_test, y_pred_oob))
            test_metrics['precision'].append(precision_score(y_test, y_pred_oob, average='weighted', zero_division=0))
            test_metrics['recall'].append(recall_score(y_test, y_pred_oob, average='weighted', zero_division=0))
            test_metrics['f1'].append(f1_score(y_test, y_pred_oob, average='weighted', zero_division=0))
            confusion_matrices.append(confusion_matrix(y_test, y_pred_oob))
            
    train_metrics_mean = {metric: np.mean(train_metrics[metric]) for metric in train_metrics}
    test_metrics_mean = {metric: np.mean(test_metrics[metric]) for metric in test_metrics}

    combined_metrics = {metric: 0.368 * train_metrics_mean[metric] + 0.632 * test_metrics_mean[metric] for metric in train_metrics}
    avg_confusion_matrix = np.mean(confusion_matrices, axis=0) if confusion_matrices else None

    combined_metrics['confusion_matrix'] = avg_confusion_matrix
    return combined_metrics

# Visualize Decision Tree (cross-validation and bootstrap)
def decision_tree_visualization(model, feature_names):
    # Visualize the Decision Tree
    buf = io.BytesIO()
    plt.figure(figsize=(100, 90))
    plot_tree(model, feature_names=feature_names, filled=True)
    plt.savefig(buf, format="png")
    plt.close()
    return base64.b64encode(buf.getbuffer()).decode("utf8")

# Evaluate models
crossval_results = {}
bootstrap_results = {}
decision_tree_cv = {}
decision_tree_bs = {}

for model_name, model_pair in models.items():
    crossval_results[model_name] = cross_validate_method(model_pair[0], X, y)
    bootstrap_results[model_name] = bootstrap_632_method(model_pair[1], X, y)


# For decision tree visualization (cross-validation and bootstrap)
decision_tree_cv = decision_tree_visualization(models["Decision Tree"][0], X.columns)
decision_tree_bootstrap = decision_tree_visualization(models["Decision Tree"][1], X.columns)

# 3. Build Dash App
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Titanic Classification Models Evaluation", style={'textAlign': 'center'}),

    # Metrics Comparison
    html.H3("Cross-Validation Metrics Comparison"),
    dcc.Graph(figure=go.Figure(
        data=[go.Bar(
            x=['accuracy', 'precision', 'recall', 'f1'],
            y=[crossval_results[model][metric] for metric in ['accuracy', 'precision', 'recall', 'f1']],
            name=model
        ) for model in models.keys()],
        layout=go.Layout(
            barmode='group',
            title="Cross-Validation Metrics Comparison",
            xaxis=dict(title="Metric"),
            yaxis=dict(title="Score")
        )
    )),
    
    html.H3("Bootstrap Metrics Comparison"),
    dcc.Graph(figure=go.Figure(
        data=[go.Bar(
            x=['accuracy', 'precision', 'recall', 'f1'],
            y=[bootstrap_results[model][metric] for metric in ['accuracy', 'precision', 'recall', 'f1']],
            name=model
        ) for model in models.keys()],
        layout=go.Layout(
            barmode='group',
            title="Bootstrap Metrics Comparison",
            xaxis=dict(title="Metric"),
            yaxis=dict(title="Score")
        )
    )),
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

    

    # Decision Tree visualizations
    html.H3("Decision Tree (Cross-Validation)"),
    html.Img(src="data:image/png;base64,{}".format(decision_tree_cv), style={'width': '100%', 'height': 'auto'}),

    html.H3("Decision Tree (Bootstrap)"),
    html.Img(src="data:image/png;base64,{}".format(decision_tree_bootstrap), style={'width': '100%', 'height': 'auto'})
])

if __name__ == '__main__':
    app.run_server(debug=True)
