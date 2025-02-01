from dash.dependencies import Input, Output
from sklearn import svm
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from helper_functions import pre_data


# 1. grid search for best hyperparameters
def grid_search(kernel):

    data = pd.read_csv("pulsar_data.csv")

    # Handle missing values
    data_cleaned = data.fillna(data.mean())

    # Separate features and target
    X = data_cleaned.drop(columns=["target_class"])
    y = data_cleaned["target_class"]

    C_range = np.logspace(-2, 2, 10)
    gamma_range = np.logspace(-2, 2, 10)
    degree_range = np.arange(2, 10)

    if kernel == 'linear':
        param_grid = {'C': C_range, 'kernel': [kernel]}
    elif kernel == 'poly':
        param_grid = {'C': C_range, 'gamma': ['scale', 'auto'], 'degree': degree_range, 'kernel': [kernel]}
    elif kernel in ['rbf', 'sigmoid']:
        param_grid = {'C': C_range, 'gamma': gamma_range, 'kernel': [kernel]}
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")
    
    cv = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
    grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2)
    grid.fit(X, y)

    print(
        "The best parameters are %s with a score of %0.2f"
        % (grid.best_params_, grid.best_score_)
    )

# 2.1 Train the model
def train_model(data, kernel, C, gamma=None, degree=None):
    '''
    Train the SVM model with the given kernel and hyperparameters.
    return: x_min, x_max, y_min, y_max, Z, svc
    x_min, x_max, y_min, y_max: The minimum and maximum values for the x and y axes
    Z: The decision function values for the meshgrid, reshaped to the meshgrid shape, used for plotting the decision boundary
    '''
    # Load the data
    X_train, y_train, X_train_pca, pca = data[0], data[2], data[4], data[6]

    # 2.1.1 Configure SVC parameters based on kernel types
    if kernel == 'linear':
        svc = svm.SVC(kernel=kernel, C=C)
    elif kernel in ['rbf', 'poly', 'sigmoid']:
        svc = svm.SVC(kernel=kernel, C=C, gamma=gamma if gamma else 'scale', degree=degree if degree else 3)
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    # 2.1.2 Train the model
    svc.fit(X_train, y_train)

    # 2.1.3 For Visualization
    # Create a meshgrid for plotting decision boundaries
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

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

    '''
    TN (True Negative): The number of instances correctly predicted as class 0.
    FP (False Positive): The number of instances incorrectly predicted as class 1 (but are actually class 0).
    FN (False Negative): The number of instances incorrectly predicted as class 0 (but are actually class 1).
    TP (True Positive): The number of instances correctly predicted as class 1.
    For example: [[2156 (TN),119 (FP)],[54 (FN),177 (TP)]]
    negative class: 0, positive class: 1
    '''
    conf_matrix = confusion_matrix(y_test, y_pred) # return [[TN, FP],[FN, TP]]

    return accuracy, precision, recall, f1, conf_matrix

# 2.3. visualize the decision boundary (comment to be added)
# def visua_decision_boundary(data, x_min, x_max, y_min, y_max, Z):

#     y_train, y_test, X_train_pca, X_test_pca = data[2], data[3], data[4], data[5]

#     # Create the decision boundary plot
#     fig = go.Figure()

#     # Add region shading to distinguish classes
#     fig.add_trace(go.Contour(
#         x=np.linspace(x_min, x_max, 200),
#         y=np.linspace(y_min, y_max, 200),
#         z=(Z > 0).astype(int),  # Shade the regions based on the decision boundary
#         colorscale=[
#             [0, "red"],
#             [1, "blue"]
#         ],  # Two distinct region colors
#         opacity=0.1,
#         showscale=False
#     ))

#     # Add scatter plot for class 0
#     fig.add_trace(go.Scatter(
#         x=X_test_pca[y_test == 0, 0],
#         y=X_test_pca[y_test == 0, 1],
#         mode='markers',
#         marker=dict(
#             color='red',
#             size=10,
#             line=dict(color='black', width=2),
#             symbol='circle-open'
#         ),
#         name='Class 0 (Non-pulsar)'
#     ))

#     # Add scatter plot for class 1
#     fig.add_trace(go.Scatter(
#         x=X_test_pca[y_test == 1, 0],
#         y=X_test_pca[y_test == 1, 1],
#         mode='markers',
#         marker=dict(
#             color='blue',
#             size=10,
#             line=dict(color='black', width=2),
#             symbol='circle-open'
#         ),
#         name='Class 1 (Pulsar)'
#     ))

#     # Add decision boundary
#     fig.add_trace(go.Contour(
#         x=np.linspace(x_min, x_max, 200),
#         y=np.linspace(y_min, y_max, 200),
#         z=Z,
#         contours=dict(
#             value=0,  # Decision boundary level
#             type="constraint",
#             showlabels=False,
#         ),
#         line=dict(color="black", width=1.5, dash="solid"),  # Solid black line
#         name="Decision Boundary",
#         legendrank=1
#     ))

#     # Add dashed lines for the margins (-0.5, 0.5)
#     fig.add_trace(go.Contour(
#         x=np.linspace(x_min, x_max, 200),
#         y=np.linspace(y_min, y_max, 200),
#         z=Z,
#         contours={"value": 0.5, "showlabels": False, "type": "constraint"},
#         line={"color": "black", "width": 1.5, "dash": "dash"},
#         name="Margin",
#         legendrank=2
#     ))
#     fig.add_trace(go.Contour(
#         x=np.linspace(x_min, x_max, 200),
#         y=np.linspace(y_min, y_max, 200),
#         z=Z,
#         contours={"value": -0.5, "showlabels": False, "type": "constraint"},
#         line={"color": "black", "width": 1.5, "dash": "dash"},
#         name="Margin",
#         showlegend=False
#     ))

#     # Add test data
#     fig.add_trace(go.Scatter(
#         x=X_test_pca[:, 0],
#         y=X_test_pca[:, 1],
#         mode='markers',
#         marker=dict(
#             color=y_test,
#             colorscale="RdBu",
#             size=10,
#             line=dict(color='black', width=2),
#             symbol='circle-open'
#         ),
#         name='Test Data',
#         showlegend=False
#     ))

#     # Add training data
#     see_training_data = 0
#     if see_training_data == 1:
#         fig.add_trace(go.Scatter(
#             x=X_train_pca[:, 0],
#             y=X_train_pca[:, 1],
#             mode='markers',
#             marker=dict(
#                 color=y_train, 
#                 colorscale='RdBu', 
#                 size=7, 
#                 line=dict(width=0.5, color='black')),
#             name='Training Data (filled circles)'
#         ))

#     # Update layout
#     fig.update_layout(
#         title="Visualization of Decision Boundaries",
#         xaxis_title="PCA Component 1",
#         yaxis_title="PCA Component 2",
#         showlegend=True
#     )
#     return fig

# 2.3. visualize the decision boundary (comment to be added)
def vis_boundary(data, svc):

    y_train, y_test, X_train_pca, X_test_pca, pca = data[2], data[3], data[4], data[5], data[6]

    # 2.1.3 For Visualization
    # Create a meshgrid for plotting decision boundaries
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    # Transform meshgrid back to original feature space
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_original = pca.inverse_transform(mesh_points)  # Project back to 8D space

    # Decision function for visualization
    Z = svc.decision_function(mesh_points_original)
    Z = Z.reshape(xx.shape)

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
        name='Class 0 (Non-pulsar)'
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
        name='Class 1 (Pulsar)'
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

# 2.4. visualize evaluation metrics figure
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


def svm_grid_train_params(data, kernel, c_range, gamma_range, degree_range):
    
    X_train, y_train = data[0], data[2]
    
    models_and_params = []
    for c in c_range:
        if kernel == 'linear':
            svc = svm.SVC(kernel='linear', C=c)
            svc.fit(X_train, y_train)
            models_and_params.append((c, svc))
        elif kernel == 'poly':
            for gamma in gamma_range:
                for degree in degree_range:
                    svc = svm.SVC(kernel='poly', C=c, gamma='auto', degree=degree)
                    svc.fit(X_train, y_train)
                    models_and_params.append((c, degree, svc))
        elif kernel in ['rbf', 'sigmoid']:
            for gamma in gamma_range:
                svc = svm.SVC(kernel=kernel, C=c, gamma=gamma)
                svc.fit(X_train, y_train)
                models_and_params.append((c, gamma, svc))
        else:
            raise ValueError(f"Unsupported kernel: {kernel}")
    
    return models_and_params




if __name__ == "__main__":

    # grid_search('linear')
    # The best parameters are {'C': np.float64(0.021544346900318832), 'kernel': 'linear'} with a score of 0.98

    grid_search('poly')



    