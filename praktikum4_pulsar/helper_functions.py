# Helper functions
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
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
        y=history['accuracy'], mode='lines+markers', name='Train Accuracy'))
    fig.add_trace(go.Scatter(
        y=history['val_accuracy'], mode='lines+markers', name='Validation Accuracy'))
    fig.add_trace(go.Scatter(
        y=history['loss'], mode='lines+markers', name='Train Loss'))
    fig.add_trace(go.Scatter(
        y=history['val_loss'], mode='lines+markers', name='Validation Loss'))
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

def block_topology(model_path, output_file):
    """
    Visualize the topology of the neural network and save as an image.
    """
    model = load_model(model_path)
    plot_model(
        model,
        to_file=output_file,
        show_shapes=True,
        show_layer_names=True,
        dpi=96
    )
    return output_file

def node_link_topology_with_neuron_weights(model_path):

    fig = go.Figure()

    # Load the model
    model = load_model(model_path)

    # Get input neurons (from model's input tensor shape)
    input_neurons = model.input_shape[-1]  # Input shape's last dimension (features)

    # Get hidden and output layers
    hidden_layers = []
    weights = []  # To store weights for connections
    for layer in model.layers:
        if hasattr(layer, "units"):  # Only Dense layers have the 'units' attribute
            hidden_layers.append(layer.units)
        if hasattr(layer, "get_weights"):  # Get weights for Dense layers
            layer_weights = layer.get_weights()
            if len(layer_weights) > 0:  # First item is the weight matrix
                weights.append(layer_weights[0])

    # Output neurons (last layer in the hidden_layers list)
    output_neurons = hidden_layers.pop()

    # Define the layers (input, hidden, output)
    layers = [input_neurons] + hidden_layers + [output_neurons]
    num_layers = len(layers)
    canvas_width = 1800  # Fixed canvas width
    canvas_height = 2500  # Fixed canvas height
    neuron_radius = 16  # Radius of the neurons for display
    layer_spacing = canvas_width // (num_layers - 1)  # Dynamically calculate horizontal spacing
    # Add nodes for each layer
    neuron_values = {}  # Store values (e.g., sum of incoming weights) for each neuron
    for layer_idx, num_nodes in enumerate(layers):
        x_position = layer_idx * layer_spacing
        # Calculate vertical offset to center neurons in the canvas
        y_offset = (canvas_height - num_nodes * 50) / 2
        for node_idx in range(num_nodes):
            y_position = y_offset + node_idx * 50

            if layer_idx == 0:  # Input layer neurons
                neuron_value = 1.0  # Placeholder value for input neurons
            else:
                # Compute the sum of weights coming into this neuron
                neuron_value = np.sum(weights[layer_idx - 1][:, node_idx])

            # Store neuron value
            neuron_values[(layer_idx, node_idx)] = neuron_value

            # Add neuron node
            fig.add_trace(go.Scatter(
                x=[x_position],
                y=[y_position],
                mode="markers+text",
                marker=dict(
                    size=neuron_radius * 2,
                    color="blue" if neuron_value >= 0 else "red",  # Blue for positive, red for negative
                ),
                text=[f"{neuron_value:.2f}"],  # Show neuron value
                textposition="middle center",
                textfont=dict(
                    size=12,        # Font size
                    color="white"   # Text color
                ),
                hoverinfo="text",
                name=f"Neuron {node_idx + 1} in Layer {layer_idx + 1}"
            ))

    # Add connections between layers
    for layer_idx, weight_matrix in enumerate(weights):
        x_start = layer_idx * layer_spacing
        x_end = (layer_idx + 1) * layer_spacing
        y_start_offset = (canvas_height - layers[layer_idx] * 50) / 2
        y_end_offset = (canvas_height - layers[layer_idx + 1] * 50) / 2
        for source_idx in range(weight_matrix.shape[0]):  # For each neuron in the current layer
            y_start = y_start_offset + source_idx * 50
            for target_idx in range(weight_matrix.shape[1]):  # For each neuron in the next layer
                y_end = y_end_offset + target_idx * 50
                weight = weight_matrix[source_idx, target_idx]
                fig.add_trace(go.Scatter(
                    x=[x_start, x_end],
                    y=[y_start, y_end],
                    mode="lines",
                    line=dict(
                        color="gray" if weight > 0 else "red",  # Gray for positive, red for negative weights
                        width=np.abs(weight) * 3  # Thickness proportional to the absolute weight value
                    ),
                    hoverinfo="text",
                    text=[f"Weight: {weight:.4f}"],
                    showlegend=False
                ))

    # Add layer labels
    for layer_idx, num_nodes in enumerate(layers):
        x_position = layer_idx * layer_spacing
        fig.add_trace(go.Scatter(
            x=[x_position],
            y=[-500],  # Position labels below the nodes
            mode="text",
            text=[f"Layer {layer_idx + 1} ({num_nodes} Neurons)"],
            textfont=dict(
                size=20,        # Font size
                color="black"   # Text color
            ),
            textposition="bottom center",
            showlegend=False
        ))

    # Adjust layout
    fig.update_layout(
        title="Neural Network Topology with Vertically Centered Neuron Weights (Full Width)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        height=canvas_height + 100,
        width=canvas_width + 200  # Add padding for better visualization
    )
    return fig


def convert_image_to_base64(image_path):
    """
    Convert an image file to base64 format for embedding in Dash.
    """
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"