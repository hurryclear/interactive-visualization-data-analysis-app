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


# 2. train the model
def train_model(kernel, C, gamma=0, degree=1): 
    # Train SVC model with a linear kernel
    svc = svm.SVC(kernel=kernel, C=C)
    svc.fit(X_train_pca, y_train)

    # Create a meshgrid for plotting decision boundaries
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    # Decision function for plotting decision boundaries
    Z = svc.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    return x_min, x_max, y_min, y_max, Z

# 3. visualize the decision boundary
def vis_decision_boundary(kernel, C, gamma, degree):
    x_min, x_max, y_min, y_max, Z = train_model(kernel, C, gamma, degree)

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
        title="Decision Boundaries of Linear Kernel in SVC",
        xaxis_title="PCA Component 1",
        yaxis_title="PCA Component 2",
        showlegend=True
    )
    return fig

# 4. evaluate the model
# def evaluate_model():

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
                min=0, max=5,  # Logical range for even spacing
                marks={0: "0", 1: "0.01", 2: "0.1", 3: "1", 4: "5", 5: "10"},
                step=None,  # Restrict slider to only these values
                value=3,  # Default value: 1 (logical position 3)
                id='c-slider'
            )
        ], style={'margin-bottom': '2px', 'width': '80%', 'margin-left': 'auto', 'margin-right': 'auto'}),  # Add spacing below the slider

        # Container for decision boundary and evaluation metrics
        html.Div([
            dcc.Graph(id='decision-boundary-linear', style={'flex': '50%', 'margin-right': '1px'}),
            dcc.Graph(id='evaluation-metrics-linear', style={'flex': '50%'})
        ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'max-width': '1500px', 'margin': 'auto'})  # Graphs side by side
    ]),

    # html.Div([
    #     # Slider for parameter 'C'
    #     html.Div([
    #         html.Label("Adjust Regularization Parameter C:"),
    #         dcc.Slider(
    #             min=0, max=5,  # Logical range for even spacing
    #             marks={0: "0", 1: "0.01", 2: "0.1", 3: "1", 4: "5", 5: "10"},
    #             step=None,  # Restrict slider to only these values
    #             value=3,  # Default value: 1 (logical position 3)
    #             id='c-slider'
    #         )
    #     ], style={'margin-bottom': '20px', 'width': '90%', 'margin-left': 'auto', 'margin-right': 'auto'}),  # Center the slider

    #     # Container for decision boundary and evaluation metrics
    #     html.Div([
    #         dcc.Graph(id='decision-boundary-linear', style={'width': '600px', 'height': '500px', 'margin-right': '20px'}),
    #         dcc.Graph(id='evaluation-metrics-linear', style={'width': '600px', 'height': '500px'})
    #     ], style={'display': 'flex', 'justify-content': 'center', 'flex-wrap': 'wrap', 'max-width': '1200px', 'margin': 'auto'})
    # ]),


    html.H2("SVM Kernel: Ploy"),
    html.Div([
        dcc.Graph(id='decision-boundary-poly'),
    ]),

    html.H2("SVM Kernel: RBF"),
    html.Div([
        dcc.Graph(id='decision-boundary-rbf'),
    ]),

    html.H2("SVM Kernel: Sigmoid"),
    html.Div([
        dcc.Graph(id='decision-boundary-sigmoid'),
    ]),
])

@app.callback(
    Output('decision-boundary-linear', 'figure'),
    Input('c-slider', 'value')
)
def update_plot(c):
    fig_linear = vis_decision_boundary("linear", C=c, gamma=0, degree=1)
    return fig_linear
    
@app.callback(
    Output('decision-boundary-poly', 'figure'),
    Input('decision-boundary-poly', 'id')  # Dummy input to trigger the initial load
)
def update_plot(_):
    fig_poly = vis_decision_boundary("poly", C=1, gamma=0, degree=1)
    return fig_poly

@app.callback(
    Output('decision-boundary-rbf', 'figure'),
    Input('decision-boundary-rbf', 'id')  # Dummy input to trigger the initial load
)
def update_plot(_):
    fig_rbf = vis_decision_boundary("rbf", C=1, gamma=0, degree=1)
    return fig_rbf

@app.callback(
    Output('decision-boundary-sigmoid', 'figure'),
    Input('decision-boundary-sigmoid', 'id')  # Dummy input to trigger the initial load
)
def update_plot(_):
    fig_rbf = vis_decision_boundary("sigmoid", C=1, gamma=0, degree=1)
    return fig_rbf




# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)