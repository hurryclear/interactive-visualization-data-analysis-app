from dash.dependencies import Input, Output
from sklearn import svm
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix



# 2.1 train the model
def train_model(data, kernel, C, gamma=None, degree=None): 
    # Load the data
    X_train, y_train, X_train_pca, pca = data[0], data[2], data[4], data[6]

    # Configure SVC parameters based on kernel type
    if kernel == 'linear':
        svc = svm.SVC(kernel=kernel, C=C)
    elif kernel in ['rbf', 'poly', 'sigmoid']:
        svc = svm.SVC(kernel=kernel, C=C, gamma=gamma if gamma else 'scale', degree=degree if degree else 3)
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    # Train the model
    svc.fit(X_train, y_train)

    # Create a meshgrid for plotting decision boundaries
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    # # Decision function for plotting decision boundaries
    # Z = svc.decision_function(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)

    # Transform meshgrid back to original feature space
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_original = pca.inverse_transform(mesh_points)  # Project back to 8D space

    # Decision function for visualization
    Z = svc.decision_function(mesh_points_original)
    Z = Z.reshape(xx.shape)

    return x_min, x_max, y_min, y_max, Z, svc

# 2.2. evaluate the model
def evaluate_model(data, svc):
    X_test, y_test = data[1], data[3]
    # Predict the test data
    y_pred = svc.predict(X_test)

    # Calculate the accuracy
    accuracy_raw = (y_pred == y_test).mean()
    # Round to 4 decimals
    accuracy = round(accuracy_raw, 4)

    # Calculate the precision, recall, and F1-score
    precision_raw = precision_score(y_test, y_pred)
    recall_raw = recall_score(y_test, y_pred)
    f1_raw = f1_score(y_test, y_pred)

    # Round each to 4 decimals
    precision = round(precision_raw, 4)
    recall = round(recall_raw, 4)
    f1 = round(f1_raw, 4)

    conf_matrix = confusion_matrix(y_test, y_pred)

    return accuracy, precision, recall, f1, conf_matrix

# 2.3. visualize the decision boundary
def visua_decision_boundary(data, x_min, x_max, y_min, y_max, Z):

    y_train, y_test, X_train_pca, X_test_pca = data[2], data[3], data[4], data[5]

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

    # Add scatter plot for class 0
    fig.add_trace(go.Scatter(
        x=X_test_pca[y_test == 0, 0],
        y=X_test_pca[y_test == 0, 1],
        mode='markers',
        marker=dict(
            color='red',
            size=10,
            line=dict(color='black', width=2),
            symbol='circle-open'
        ),
        name='Class 0'
    ))

    # Add scatter plot for class 1
    fig.add_trace(go.Scatter(
        x=X_test_pca[y_test == 1, 0],
        y=X_test_pca[y_test == 1, 1],
        mode='markers',
        marker=dict(
            color='blue',
            size=10,
            line=dict(color='black', width=2),
            symbol='circle-open'
        ),
        name='Class 1'
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
        name="Decision Boundary",
        legendrank=1
    ))

    # Add dashed lines for the margins (-0.5, 0.5)
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 200),
        y=np.linspace(y_min, y_max, 200),
        z=Z,
        contours={"value": 0.5, "showlabels": False, "type": "constraint"},
        line={"color": "black", "width": 1.5, "dash": "dash"},
        name="Margin",
        legendrank=2
    ))
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 200),
        y=np.linspace(y_min, y_max, 200),
        z=Z,
        contours={"value": -0.5, "showlabels": False, "type": "constraint"},
        line={"color": "black", "width": 1.5, "dash": "dash"},
        name="Margin",
        showlegend=False
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
        name='Test Data',
        showlegend=False
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
def evaluation_metrics(accuracy, precision, recall, f1):
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
        xaxis=dict(title='Metrics'),
        showlegend=False
    )
    return fig

