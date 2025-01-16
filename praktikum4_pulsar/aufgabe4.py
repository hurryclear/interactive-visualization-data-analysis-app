import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import precision_score, recall_score, f1_score


# 1. prepare the data (load, clean, split, standardize, pca)
# Load the dataset
data = pd.read_csv('pulsar_data.csv')

# Handle missing values by replacing them with the mean of the column
data_cleaned = data.fillna(data.mean())

# Feature X and target y
X = data_cleaned.drop(columns=["target_class"])
y = data_cleaned["target_class"]

# Split the data into training and test data (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reduce data to 2D for visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 2.1 train the model
def train_model(kernel, C, gamma=None, degree=None): 
    # Configure SVC parameters based on kernel type
    if kernel == 'linear':
        svc = svm.SVC(kernel=kernel, C=C)
    elif kernel in ['rbf', 'poly', 'sigmoid']:
        svc = svm.SVC(kernel=kernel, C=C, gamma=gamma if gamma else 'scale', degree=degree if degree else 3)
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    # Train the model
    svc.fit(X_train_pca, y_train)

    # Create a meshgrid for plotting decision boundaries
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    # Decision function for plotting decision boundaries
    Z = svc.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    return x_min, x_max, y_min, y_max, Z, svc

# 2.2. evaluate the model
def evaluate_model(svc, X_test_pca, y_test):
    # Predict the test data
    y_pred = svc.predict(X_test_pca)

    # Calculate the accuracy
    accuracy = (y_pred == y_test).mean()

    # Calculate the precision, recall, and F1-score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, precision, recall, f1

# 2.3. visualize the decision boundary
def visua_decision_boundary(x_min, x_max, y_min, y_max, Z):

    # Create the decision boundary plot
    fig = go.Figure()

    # Add region shading to distinguish classes
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 200),
        y=np.linspace(y_min, y_max, 200),
        z=(Z > 0).astype(int),  # Shade the regions based on the decision boundary
        colorscale=[
            [0, "red"], 
            [1, "blue"]
        ],  # Two distinct region colors
        opacity=0.1,
        showscale=False
    ))

    # Add decision boundary
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 200),
        y=np.linspace(y_min, y_max, 200),
        z=Z,
        contours=dict(
            value=0,  # Decision boundary level
            type="constraint",
            showlabels=False,
        ),
        line=dict(color="black", width=1.5, dash="solid"),  # Solid black line
        name="Decision Boundary"
    ))

    # Add dashed lines for the margins (-0.5, 0.5)
    for margin in [-0.5, 0.5]:
        fig.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, 200),
            y=np.linspace(y_min, y_max, 200),
            z=Z,
            contours=dict(
                value=margin,  # Margin level
                showlabels=False,
                type="constraint"
            ),
            line=dict(color="black", width=1.5, dash="dash"),
            name="Margin"
        ))

    # Add test data
    fig.add_trace(go.Scatter(
        x=X_test_pca[:, 0],
        y=X_test_pca[:, 1],
        mode='markers',
        marker=dict(
            color=y_test,
            colorscale="RdBu",
            size=10,
            line=dict(color='black', width=2),
            symbol='circle-open'
        ),
        name='Test Data'
    ))

    # Add training data
    see_training_data = 0
    if see_training_data == 1:
        fig.add_trace(go.Scatter(
            x=X_train_pca[:, 0],
            y=X_train_pca[:, 1],
            mode='markers',
            marker=dict(
                color=y_train, 
                colorscale='RdBu', 
                size=7, 
                line=dict(width=0.5, color='black')),
            name='Training Data (filled circles)'
        ))

    # Update layout
    fig.update_layout(
        title="Visualization of Decision Boundaries",
        xaxis_title="PCA Component 1",
        yaxis_title="PCA Component 2",
        showlegend=True
    )
    return fig

# 2.4. generate evaluation metrics figure
def visua_evaluation(accuracy, precision, recall, f1):
    fig = go.Figure(data=[
        go.Bar(name='Accuracy', x=['Accuracy'], y=[accuracy], text=[f"{accuracy:.4f}"], textposition='outside'),
        go.Bar(name='Precision', x=['Precision'], y=[precision], text=[f"{precision:.4f}"], textposition='outside'),
        go.Bar(name='Recall', x=['Recall'], y=[recall], text=[f"{recall:.4f}"], textposition='outside'),
        go.Bar(name='F1-Score', x=['F1-Score'], y=[f1], text=[f"{f1:.4f}"], textposition='outside')
    ])
    fig.update_layout(
        barmode='group',
        title='Evaluation Metrics',
        yaxis=dict(title='Score', range=[0, 1.1]),  # Adjust range to accommodate text above bars
        xaxis=dict(title='Metrics')
    )
    return fig

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("SVM Decision Boundary Visualization"),

    html.H2("SVM Kernel: Linear"),
    html.Div([
        # Slider for parameter 'C'
        html.Div([
            html.Label("Adjust Regularization Parameter C:"),
            dcc.Slider(
                min=0, max=4,  # Logical range for even spacing
                marks={0: "0.01", 1: "0.1", 2: "1", 3: "5", 4: "10"},
                step=None,  # Restrict slider to only these values
                value=3,  # Default value: 1 (logical position 3)
                id='c-slider-linear'
            )
        ], style={'margin-bottom': '2px', 'width': '80%', 'margin-left': 'auto', 'margin-right': 'auto'}),  # Add spacing below the slider

        # Container for decision boundary and evaluation metrics
        html.Div([
            dcc.Graph(id='decision-boundary-linear', style={'flex': '50%', 'margin-right': '1px'}),
            dcc.Graph(id='evaluation-metrics-linear', style={'flex': '50%'})
        ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'max-width': '1500px', 'margin': 'auto'})  # Graphs side by side
    ]),

    html.H2("SVM Kernel: Poly"),
    html.Div([
        # Slider for parameter 'C'
        html.Div([
            html.Label("Adjust Regularization Parameter C:"),
            dcc.Slider(
                min=0, max=4,  # Logical range for even spacing
                marks={0: "0.01", 1: "0.1", 2: "1", 3: "5", 4: "10"},
                step=None,  # Restrict slider to only these values
                value=3,  # Default value: 1 (logical position 3)
                id='c-slider-poly'
            )
        ], style={'margin-bottom': '2px', 'width': '80%', 'margin-left': 'auto', 'margin-right': 'auto'}),  # Add spacing below the slider

        html.Div([
            html.Label("Adjust Degree:"),
            dcc.Slider(
                min=0, max=8,  # Logical range for even spacing
                marks={0: "2", 1: "3", 2: "4", 3: "5", 4: "6", 5: "7", 6: "8", 7: "9", 8: "10"},  # Define degree options
                step=None,  # Restrict slider to only these values
                value=1,  # Default value: 3
                id='degree-slider-poly'
            )
        ], style={'margin-bottom': '2px', 'width': '80%', 'margin-left': 'auto', 'margin-right': 'auto'}),

        # Container for decision boundary and evaluation metrics
        html.Div([
            dcc.Graph(id='decision-boundary-poly', style={'flex': '50%', 'margin-right': '1px'}),
            dcc.Graph(id='evaluation-metrics-poly', style={'flex': '50%'})
        ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'max-width': '1500px', 'margin': 'auto'})  # Graphs side by side
    ]),

    html.H2("SVM Kernel: RBF"),
    html.Div([
        # Slider for parameter 'C'
        html.Div([
            html.Label("Adjust Regularization Parameter C:"),
            dcc.Slider(
                min=0, max=5,  # Logical range for even spacing
                marks={0: "0.01", 1: "0.1", 2: "1", 3: "5", 4: "10"},
                step=None,  # Restrict slider to only these values
                value=3,  # Default value: 1 (logical position 3)
                id='c-slider-rbf'
            )
        ], style={'margin-bottom': '2px', 'width': '80%', 'margin-left': 'auto', 'margin-right': 'auto'}),  # Add spacing below the slider
        html.Div([
            html.Label("Adjust Gamma:"),
            dcc.Slider(
                min=0, max=3,  # Logical range for even spacing
                marks={0: "0.1", 1: "1", 2: "5", 3: "10"},
                step=None,  # Restrict slider to only these values
                value=2,  # Default value: 1
                id='gamma-slider-rbf'
            )
        ], style={'margin-bottom': '2px', 'width': '80%', 'margin-left': 'auto', 'margin-right': 'auto'}),

        # Container for decision boundary and evaluation metrics
        html.Div([
            dcc.Graph(id='decision-boundary-rbf', style={'flex': '50%', 'margin-right': '1px'}),
            dcc.Graph(id='evaluation-metrics-rbf', style={'flex': '50%'})
        ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'max-width': '1500px', 'margin': 'auto'})  # Graphs side by side
    ]),

    html.H2("SVM Kernel: Sigmoid"),
    html.Div([
        # Slider for parameter 'C'
        html.Div([
            html.Label("Adjust Regularization Parameter C:"),
            dcc.Slider(
                min=0, max=4,  # Logical range for even spacing
                marks={0: "0.01", 1: "0.1", 2: "1", 3: "5", 4: "10"},
                step=None,  # Restrict slider to only these values
                value=3,  # Default value: 1 (logical position 3)
                id='c-slider-sigmoid'
            )
        ], style={'margin-bottom': '2px', 'width': '80%', 'margin-left': 'auto', 'margin-right': 'auto'}),  # Add spacing below the slider

        # Container for decision boundary and evaluation metrics
        html.Div([
            dcc.Graph(id='decision-boundary-sigmoid', style={'flex': '50%', 'margin-right': '1px'}),
            dcc.Graph(id='evaluation-metrics-sigmoid', style={'flex': '50%'})
        ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'max-width': '1500px', 'margin': 'auto'})  # Graphs side by side
    ]),
])

@app.callback(
    [Output('decision-boundary-linear', 'figure'),
    Output('evaluation-metrics-linear', 'figure')],
    [Input('c-slider-linear', 'value')]
)
def update_plot(c_position):
    # Map slider position to actual C values
    c_values = [0.01, 0.1, 1, 5, 10]
    c = c_values[int(c_position)]

    x_min, x_max, y_min, y_max, Z, svc = train_model("linear", c)
    accuracy, precision, recall, f1 = evaluate_model(svc, X_test_pca, y_test)
    decision_boundary_linear = visua_decision_boundary(x_min, x_max, y_min, y_max, Z)
    evaluation_metrics_linear = visua_evaluation(accuracy, precision, recall, f1)
    return decision_boundary_linear, evaluation_metrics_linear

@app.callback(
    [Output('decision-boundary-poly', 'figure'),
    Output('evaluation-metrics-poly', 'figure')],
    [Input('c-slider-poly', 'value'),
    Input('degree-slider-poly', 'value')]
)
def update_plot(c_position, degree_position):
    # Map slider positions to actual C and degree values
    c_values = [0.01, 0.1, 1, 5, 10]
    degree_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]  # Define degree options
    c = c_values[int(c_position)]
    degree = degree_values[int(degree_position)]

    x_min, x_max, y_min, y_max, Z, svc = train_model("poly", c, degree=degree) # degree should be changable
    accuracy, precision, recall, f1 = evaluate_model(svc, X_test_pca, y_test)
    decision_boundary_poly = visua_decision_boundary(x_min, x_max, y_min, y_max, Z)
    evaluation_metrics_poly = visua_evaluation(accuracy, precision, recall, f1)
    return decision_boundary_poly, evaluation_metrics_poly

@app.callback(
    [Output('decision-boundary-rbf', 'figure'),
    Output('evaluation-metrics-rbf', 'figure')],
    [Input('c-slider-rbf', 'value'),
    Input('gamma-slider-rbf', 'value')]
)
def update_plot(c_position, gamma_position):
    # Map slider positions to actual C and gamma values
    c_values = [0.01, 0.1, 1, 5, 10]
    gamma_values = [0.1, 1, 5, 10]
    c = c_values[int(c_position)]
    gamma = gamma_values[int(gamma_position)]

    x_min, x_max, y_min, y_max, Z, svc = train_model("rbf", c, gamma=gamma) # gamma should be changable
    accuracy, precision, recall, f1 = evaluate_model(svc, X_test_pca, y_test)
    decision_boundary_rbf = visua_decision_boundary(x_min, x_max, y_min, y_max, Z)
    evaluation_metrics_rbf = visua_evaluation(accuracy, precision, recall, f1)
    return decision_boundary_rbf, evaluation_metrics_rbf

@app.callback(
    [Output('decision-boundary-sigmoid', 'figure'),
    Output('evaluation-metrics-sigmoid', 'figure')],
    [Input('c-slider-sigmoid', 'value')]
)
def update_plot(c_position):
    c_values = [0.01, 0.1, 1, 5, 10]
    c = c_values[int(c_position)]
    x_min, x_max, y_min, y_max, Z, svc = train_model("sigmoid", c, gamma=2, degree=3)
    accuracy, precision, recall, f1 = evaluate_model(svc, X_test_pca, y_test)
    decision_boundary_sigmoid = visua_decision_boundary(x_min, x_max, y_min, y_max, Z)
    evaluation_metrics_sigmoid = visua_evaluation(accuracy, precision, recall, f1)
    return decision_boundary_sigmoid, evaluation_metrics_sigmoid



# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)