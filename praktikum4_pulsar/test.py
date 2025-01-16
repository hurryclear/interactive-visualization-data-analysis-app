from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import numpy as np
from sklearn import datasets, svm
import plotly.graph_objects as go

# Load and preprocess the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Use only two classes and two features for visualization
X = X[y != 0, :2]
y = y[y != 0]

# Shuffle the dataset
np.random.seed(0)
order = np.random.permutation(len(X))
X = X[order]
y = y[order].astype(float)

# Split into training and testing sets
n_sample = len(X)
X_train = X[:int(0.9 * n_sample)]
y_train = y[:int(0.9 * n_sample)]
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
    dcc.Graph(id='svm-plot', style={"height": "80vh"})
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

    # Create the figure
    fig = go.Figure()

    # Add the decision boundary and margins
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 200),
        y=np.linspace(y_min, y_max, 200),
        z=Z,
        showscale=False,
        colorscale='RdBu',
        opacity=0.8,
        contours=dict(
            start=-1, end=1, size=0.5,
            coloring="lines"
        )
    ))

    # Add training data points
    fig.add_trace(go.Scatter(
        x=X_train[:, 0],
        y=X_train[:, 1],
        mode='markers',
        marker=dict(color=y_train, colorscale='RdBu', size=8, line=dict(color='black', width=1)),
        name='Training Data'
    ))

    # Add test data points (circled for distinction)
    fig.add_trace(go.Scatter(
        x=X_test[:, 0],
        y=X_test[:, 1],
        mode='markers',
        marker=dict(
            color=y_test,
            colorscale='RdBu',
            size=12,
            line=dict(color='black', width=2),
            symbol='circle-open'
        ),
        name='Test Data'
    ))

    # Update layout
    fig.update_layout(
        title=f"SVM Decision Boundary with {kernel} Kernel",
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max]),
        height=700
    )

    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)