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
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train SVC model with a linear kernel
svc = svm.SVC(kernel="linear", C=1)
svc.fit(X_train_pca, y_train)

# Create a meshgrid for plotting decision boundaries
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# Decision function for plotting decision boundaries
Z = svc.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Improved SVC Decision Boundary Visualization"),
    dcc.Graph(id='decision-boundary-plot'),
])

@app.callback(
    Output('decision-boundary-plot', 'figure'),
    Input('decision-boundary-plot', 'id')  # Dummy input to trigger the initial load
)
def update_decision_boundary(_):
    # Create the decision boundary plot
    fig = go.Figure()

    # Add background colors for the regions
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 200),
        y=np.linspace(y_min, y_max, 200),
        z=Z,
        showscale=False,
        colorscale=[(0, "purple"), (0.5, "purple"), (0.5, "yellow"), (1, "yellow")],
        opacity=0.4,
        contours=dict(
            start=-1,
            end=1,
            size=2,
            coloring="fill",
        )
    ))

    # Add decision boundary and margins
    fig.add_contour(
        x=np.linspace(x_min, x_max, 200),
        y=np.linspace(y_min, y_max, 200),
        z=Z,
        contours=dict(
            start=-1, end=1, size=1,
            coloring="lines",
        ),
        line_width=2,
        line_color="black",
        line_dash="solid"
    )

    # # Add decision boundary
    # fig.add_contour(
    #     x=np.linspace(x_min, x_max, 200),
    #     y=np.linspace(y_min, y_max, 200),
    #     z=Z,
    #     contours=dict(
    #         start=0, end=0, size=1,
    #     ),
    #     line_width=2,
    #     line_color="black",
    #     line_dash="solid"
    # )

    # # Add margin boundaries
    # fig.add_contour(
    #     x=np.linspace(x_min, x_max, 200),
    #     y=np.linspace(y_min, y_max, 200),
    #     z=Z,
    #     contours=dict(
    #         start=-1, end=1, size=2,
    #     ),
    #     line_width=2,
    #     line_color="black",
    #     line_dash="dash"
    # )


    # Add training samples
    fig.add_trace(go.Scatter(
        x=X_train_pca[:, 0],
        y=X_train_pca[:, 1],
        mode='markers',
        marker=dict(color=y_train, colorscale='Viridis', size=7, line=dict(width=0.5, color='black')),
        name='Training Samples'
    ))

    # Highlight support vectors
    fig.add_trace(go.Scatter(
        x=svc.support_vectors_[:, 0],
        y=svc.support_vectors_[:, 1],
        mode='markers',
        marker=dict(color='black', size=9, symbol='circle-open'),
        name='Support Vectors'
    ))

    # Update layout
    fig.update_layout(
        title="Decision Boundaries of Linear Kernel in SVC",
        xaxis_title="PCA Component 1",
        yaxis_title="PCA Component 2",
        showlegend=True
    )
    return fig

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)