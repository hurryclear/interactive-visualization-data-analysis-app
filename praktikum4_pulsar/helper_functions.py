# Helper functions
import pandas as pd
import numpy as np
import base64
from joblib import load
import plotly.graph_objects as go
from tensorflow.keras.utils import plot_model
# from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pre_data(pca_components=None):
    """
    Prepare the dataset: load, clean, split, and preprocess.

    Parameters:
        data_path (str): Path to the dataset.
        pca_components (int): Number of PCA components for dimensionality reduction. Default is None.

    Returns:
        X_train (np.array): Preprocessed training features.
        X_test (np.array): Preprocessed test features.
        y_train (np.array): Training labels.
        y_test (np.array): Test labels.
        X_train_pca (np.array): PCA-transformed training features (if PCA applied).
        X_test_pca (np.array): PCA-transformed test features (if PCA applied).
    """
    # Load the dataset
    data = pd.read_csv("pulsar_data.csv")

    # Handle missing values
    data_cleaned = data.fillna(data.mean())

    # Separate features and target
    X = data_cleaned.drop(columns=["target_class"])
    y = data_cleaned["target_class"]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Optional: Apply PCA
    if pca_components:
        pca = PCA(n_components=pca_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
    else:
        X_train_pca = None
        X_test_pca = None

    return [X_train, X_test, y_train, y_test, X_train_pca, X_test_pca, pca]

def calculate_accuracy(confusion_matrix):
    """
    Calculate accuracy from the confusion matrix.
    """
    true_positives_and_negatives = sum(confusion_matrix[i][i] for i in range(len(confusion_matrix)))
    total_samples = sum(sum(row) for row in confusion_matrix)
    accuracy = true_positives_and_negatives / total_samples
    return accuracy

def learning_curves_dff(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=history['loss_curve'], mode='lines+markers', name='loss_curve'))
    fig.add_trace(go.Scatter(
        y=history['validation_scores'], mode='lines+markers', name='validation_scores'))
    # fig.add_trace(go.Scatter(
    #     y=history['accuracy'], mode='lines+markers', name='Train Accuracy'))
    # fig.add_trace(go.Scatter(
    #     y=history['val_accuracy'], mode='lines+markers', name='Validation Accuracy'))
    # fig.add_trace(go.Scatter(
    #     y=history['loss'], mode='lines+markers', name='Train Loss'))
    # fig.add_trace(go.Scatter(
    #     y=history['val_loss'], mode='lines+markers', name='Validation Loss'))
    fig.update_layout(
        title="Learning Curves",
        xaxis_title="Epochs",
        yaxis_title="Metrics",
        legend_title="Metrics"
    )
    return fig

def confusion_matrix_dff(conf_matrix):
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=["Predicted 0", "Predicted 1"],
        y=["Actual 0", "Actual 1"],
        colorscale="Blues",
        showscale=True,
        text=conf_matrix,
        texttemplate="%{text}"
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )
    return fig



def node_link_topology_with_neuron_weights(model_path):
    """
    Visualize an sklearn MLPClassifier's topology and weights using Plotly.
    """

    # 1. Load the sklearn MLP model
    model = load(model_path)
    weights = model.coefs_

    # 2. Determine layer sizes
    #    coefs_[0]: (n_features, hidden_1)
    #    ...
    #    coefs_[-1]: (hidden_last, n_outputs)
    input_size = weights[0].shape[0]
    hidden_sizes = [w.shape[1] for w in weights[:-1]]
    output_size = weights[-1].shape[1]
    layers = [input_size] + hidden_sizes + [output_size]
    num_layers = len(layers)

    # 3. Canvas configuration
    canvas_width = 1800
    canvas_height = 2500
    margin = 100
    neuron_radius = 16
    layer_spacing = (canvas_width // (num_layers - 1)) if num_layers > 1 else 300

    # 4. Create the figure
    fig = go.Figure()

    def get_y_positions(num_nodes):
        """
        Positions nodes evenly (top to bottom).
        Returns a list of float positions in [margin, canvas_height - margin].
        """
        if num_nodes == 1:
            return [float(canvas_height / 2)]
        return [float(y) for y in np.linspace(margin, canvas_height - margin, num_nodes)]

    # 5. Plot neurons (Scatter traces with mode="markers+text")
    for layer_idx, num_nodes in enumerate(layers):
        x_coord = layer_idx * layer_spacing
        y_positions = get_y_positions(num_nodes)

        for node_idx, y_coord in enumerate(y_positions):
            # For input layer, no incoming weights to sum
            if layer_idx == 0:
                neuron_value = 0.0
            else:
                # Sum the weights from previous layer to this neuron
                w_matrix = weights[layer_idx - 1]
                neuron_value = float(np.sum(w_matrix[:, node_idx]))  # ensure native float

            # Color code based on sign
            color = "blue" if neuron_value >= 0 else "red"

            # Add a single scatter point for each neuron
            fig.add_trace(go.Scatter(
                x=[x_coord],
                y=[y_coord],
                mode="markers+text",
                marker=dict(size=neuron_radius * 2, color=color),
                text=[f"{neuron_value:.2f}"],  # 1-element list matching x/y length
                textposition="middle center",
                textfont=dict(size=12, color="white"),
                hoverinfo="text",   # what shows on hover
                hovertext=f"Layer {layer_idx+1}, Neuron {node_idx+1}",
                showlegend=False    # hide from legend
            ))

    # 6. Plot connections (Scatter traces with mode="lines")
    for layer_idx in range(len(weights)):
        w_matrix = weights[layer_idx]  # shape: (layers[layer_idx], layers[layer_idx+1])
        x_start = layer_idx * layer_spacing
        x_end = (layer_idx + 1) * layer_spacing

        source_positions = get_y_positions(layers[layer_idx])
        target_positions = get_y_positions(layers[layer_idx + 1])

        for src_idx in range(w_matrix.shape[0]):
            for tgt_idx in range(w_matrix.shape[1]):
                weight = float(w_matrix[src_idx, tgt_idx])
                line_color = "gray" if weight >= 0 else "red"
                fig.add_trace(go.Scatter(
                    x=[x_start, x_end],
                    y=[source_positions[src_idx], target_positions[tgt_idx]],
                    mode="lines",
                    line=dict(color=line_color, width=abs(weight) * 3),
                    hoverinfo="text",
                    hovertext=f"Weight: {weight:.4f}",
                    showlegend=False
                ))

    # 7. Add layer labels as annotations
    for layer_idx, num_nodes in enumerate(layers):
        fig.add_annotation(
            x=layer_idx * layer_spacing,
            y=-100,  # place below the neurons
            text=f"Layer {layer_idx+1}<br>({num_nodes} neurons)",
            showarrow=False,
            yanchor="top",
            font=dict(size=14)
        )

    # 8. Update the layout
    fig.update_layout(
        title="MLP Classifier Topology",
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-100, canvas_width + 100]
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-200, canvas_height + 100],
            # 'scaleanchor': 'x'  # If you really want to fix the aspect ratio, uncomment
        ),
        height=canvas_height + 200,
        width=canvas_width + 200,
        margin=dict(b=150),  # space for the layer labels
        plot_bgcolor="white"
    )

    # 9. Return the final figure
    return fig


def convert_image_to_base64(image_path):
    """
    Convert an image file to base64 format for embedding in Dash.
    """
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"