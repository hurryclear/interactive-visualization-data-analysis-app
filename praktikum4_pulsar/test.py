from dash import Dash, dcc, html, Input, Output
import numpy as np
from sklearn import svm
import plotly.graph_objects as go
import pandas as pd

# Load the dataset
file_path = 'pulsar_data.csv'  # Update with your file's path
pulsar_data = pd.read_csv(file_path)

# Select two features and the target class for visualization
selected_features = ['Mean of the integrated profile', 'Standard deviation of the integrated profile']
target_column = 'target_class'

# Filter the data
data = pulsar_data.dropna()  # Handle any NaN values
X = data[selected_features].values
y = data[target_column].values

# Filter out one class for a binary classification setup as in the original code
X = X[y != 0]
y = y[y != 0]

# Shuffle and split the data into training and testing sets
np.random.seed(0)
order = np.random.permutation(len(X))
X = X[order]
y = y[order].astype(float)

n_sample = len(X)
X_train = X[: int(0.9 * n_sample)]
y_train = y[: int(0.9 * n_sample)]
X_test = X[int(0.9 * n_sample):]
y_test = y[int(0.9 * n_sample):]

# Initialize the Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Interactive SVM Decision Boundary Visualization"),
    dcc.Dropdown(
        id='kernel-selector',
        options=[
            {'label': 'Linear', 'value': 'linear'},
            {'label': 'RBF', 'value': 'rbf'},
            {'label': 'Polynomial', 'value': 'poly'}
        ],
        value='linear',
        clearable=False
    ),
    dcc.Graph(id='svm-plot')
])

@app.callback(
    Output('svm-plot', 'figure'),
    [Input('kernel-selector', 'value')]
)
def update_plot(kernel):
    # Train the SVM model with the selected kernel
    clf = svm.SVC(kernel=kernel, gamma=10)
    clf.fit(X_train, y_train)
    
    # Create a mesh for decision boundary visualization
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    XX, YY = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)
    
    # Create the scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X[:, 0], y=X[:, 1],
        mode='markers',
        marker=dict(color=y, colorscale='Viridis', line=dict(width=1, color='Black')),
        name='Data Points'
    ))
    fig.add_trace(go.Scatter(
        x=X_test[:, 0], y=X_test[:, 1],
        mode='markers',
        marker=dict(color='rgba(255,255,255,0)', line=dict(width=2, color='Black')),
        name='Test Points'
    ))
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 200),
        y=np.linspace(y_min, y_max, 200),
        z=Z,
        colorscale='Viridis',
        opacity=0.5,
        showscale=False,
        contours=dict(showlines=False)
    ))
    fig.update_layout(
        title=f"SVM with {kernel} kernel",
        xaxis_title=selected_features[0],
        yaxis_title=selected_features[1],
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)