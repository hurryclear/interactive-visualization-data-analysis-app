import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], errors='ignore')
df['Age'] = np.floor(df['Age'])
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
X = df.drop(columns=['Survived'])
y = df['Survived']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
}

# Cross-validation method
def cross_validate_method(model, X, y):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    confusion_matrices = []

    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
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
    }

# Bootstrap method
def bootstrap_632_method(model, X, y):
    train_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    test_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    confusion_matrices = []

    for _ in range(100):  
        X_boot, y_boot = resample(X, y, replace=True, n_samples=len(X), random_state=42)
        oob_indices = np.setdiff1d(np.arange(len(X)), np.unique(X_boot, return_index=True)[1])
        X_oob, y_oob = X[oob_indices], y[oob_indices]

        model.fit(X_boot, y_boot)
        y_pred_train = model.predict(X_boot)

        train_metrics['accuracy'].append(accuracy_score(y_boot, y_pred_train))
        train_metrics['precision'].append(precision_score(y_boot, y_pred_train, average='weighted', zero_division=0))
        train_metrics['recall'].append(recall_score(y_boot, y_pred_train, average='weighted', zero_division=0))
        train_metrics['f1'].append(f1_score(y_boot, y_pred_train, average='weighted', zero_division=0))

        if len(oob_indices) > 0:
            y_pred_oob = model.predict(X_oob)
            test_metrics['accuracy'].append(accuracy_score(y_oob, y_pred_oob))
            test_metrics['precision'].append(precision_score(y_oob, y_pred_oob, average='weighted', zero_division=0))
            test_metrics['recall'].append(recall_score(y_oob, y_pred_oob, average='weighted', zero_division=0))
            test_metrics['f1'].append(f1_score(y_oob, y_pred_oob, average='weighted', zero_division=0))
            confusion_matrices.append(confusion_matrix(y_oob, y_pred_oob))
            
    train_metrics_mean = {metric: np.mean(train_metrics[metric]) for metric in train_metrics}
    test_metrics_mean = {metric: np.mean(test_metrics[metric]) for metric in test_metrics}

    combined_metrics = {metric: 0.368 * train_metrics_mean[metric] + 0.632 * test_metrics_mean[metric] for metric in train_metrics}
    avg_confusion_matrix = np.mean(confusion_matrices, axis=0) if confusion_matrices else None

    combined_metrics['confusion_matrix'] = avg_confusion_matrix
    return combined_metrics

# Evaluate models
crossval_results = {}
bootstrap_results = {}

for model_name, model in models.items():
    crossval_results[model_name] = cross_validate_method(model, X_scaled, y)
    bootstrap_results[model_name] = bootstrap_632_method(model, X_scaled, y)

# decision tree visualisieren
def decision_tree_cv_visualization(model, X, y):
    # Cross-Validation: Trainiere den Baum auf den gesamten Trainingsdaten der Cross-Validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # Wir trainieren den Baum auf den gesamten Trainingsdaten
    for train_idx, test_idx in cv.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        model.fit(X_train, y_train)
    
    # Visualisiere den Baum nach der gesamten Cross-Validation
    buf = io.BytesIO()
    plt.figure(figsize=(100, 90))
    plot_tree(model, feature_names=X.columns, class_names=["Not Survived", "Survived"], filled=True)
    plt.savefig(buf, format="png")
    plt.close()
    return base64.b64encode(buf.getbuffer()).decode("utf8")


def decision_tree_bootstrap_visualization(model, X, y):
    X_boot, y_boot = resample(X, y, replace=True, n_samples=len(X), random_state=42)
    model.fit(X_boot, y_boot)
    
    # Visualisiere den Baum
    buf = io.BytesIO()
    plt.figure(figsize=(100, 90))
    plot_tree(model, feature_names=X.columns, class_names=["Not Survived", "Survived"], filled=True)
    plt.savefig(buf, format="png")
    plt.close()
    return base64.b64encode(buf.getbuffer()).decode("utf8")


decision_tree_cv = decision_tree_cv_visualization(models["Decision Tree"], X, y)
decision_tree_bootstrap = decision_tree_bootstrap_visualization(models["Decision Tree"], X, y)


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

    html.H3("Decision Tree (Bootstrap)"),
    html.Img(src="data:image/png;base64,{}".format(decision_tree_bootstrap), style={'width': '100%', 'height': 'auto'})
])

if __name__ == '__main__':
    app.run_server(debug=True)



    