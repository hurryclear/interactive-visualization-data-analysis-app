# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

# 1. DATA PREPARATION
df = pd.read_csv('titanic.csv')

# Clean the data
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], errors='ignore')
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Features and target
X = df.drop(columns=['Survived'])
y = df['Survived']

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3)
}

# 2. Helper Functions
def cross_validate_model(model, X, y):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores

def bootstrap_evaluation(model, X, y):
    accuracies = []
    for i in range(100):
        X_resampled, y_resampled = resample(X, y, replace=True)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    return np.mean(accuracies)

# Prepare results
results = {}
for model_name, model in models.items():
    results[model_name] = {
        'cross_val': cross_validate_model(model, X, y),
        'bootstrap': bootstrap_evaluation(model, X, y)
    }

# 3. Build Dash App
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Titanic Classification Task", style={'textAlign': 'center'}),

    # Dropdown to select model
    html.Div([
        html.Label("Select Classifier:"),
        dcc.Dropdown(
            id='model-dropdown',
            options=[{'label': name, 'value': name} for name in models.keys()],
            value='Logistic Regression'
        )
    ], style={'width': '50%', 'margin': 'auto'}),

    # Cross-validation plot
    html.Div([
        dcc.Graph(id='crossval-plot')
    ]),

    # Bootstrap accuracy
    html.Div([
        dcc.Graph(id='bootstrap-plot')
    ]),

    # Confusion matrix
    html.Div([
        dcc.Graph(id='confusion-matrix')
    ])
])

# Callbacks
@app.callback(
    [Output('crossval-plot', 'figure'),
     Output('bootstrap-plot', 'figure'),
     Output('confusion-matrix', 'figure')],
    Input('model-dropdown', 'value')
)
def update_plots(selected_model):
    model = models[selected_model]
    
    # Cross-validation results
    cross_val_scores = results[selected_model]['cross_val']
    crossval_fig = px.line(
        y=cross_val_scores, 
        x=list(range(1, 11)),
        title=f"Cross-Validation Scores for {selected_model}",
        labels={'x': 'Fold', 'y': 'Accuracy'}
    )

    # Bootstrap accuracy
    bootstrap_accuracy = results[selected_model]['bootstrap']
    bootstrap_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bootstrap_accuracy,
        title={'text': f"Bootstrap Accuracy (0.632 Method) for {selected_model}"}
    ))

    # Confusion matrix
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_fig = px.imshow(
        cm, text_auto=True, color_continuous_scale='Blues',
        labels=dict(x="Predicted", y="Actual", color="Count"),
        title=f"Confusion Matrix for {selected_model}"
    )
    cm_fig.update_xaxes(side="top")

    return crossval_fig, bootstrap_fig, cm_fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)