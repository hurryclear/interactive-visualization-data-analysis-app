# Helper functions
import pandas as pd
import numpy as np
import base64
from joblib import load
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pre_data(pca_components=None):
    """
    Prepare the dataset: load, clean, split, standardize, and optionally apply PCA.

    Parameters:
        pca_components (int): Number of PCA components to use (if any).

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
        showscale=False,
        text=conf_matrix,
        texttemplate="%{text}"
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )
    return fig

def build_line_diagram(all_metrics):
    # Convert to sorted list of C values
    sorted_cs = sorted(all_metrics.keys())
    
    # Extract metrics in order
    accuracy_vals = []
    precision_vals = []
    recall_vals = []
    f1_vals = []

    for c in sorted_cs:
        metrics = all_metrics[c]
        accuracy_vals.append(metrics["accuracy"])
        precision_vals.append(metrics["precision"])
        recall_vals.append(metrics["recall"])
        f1_vals.append(metrics["f1"])

    # Create figure
    fig = go.Figure()
    
    # Helper function to add traces
    def add_trace(x, y, name):
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            name=name,
            hovertemplate=f"<b>{name}</b><br>C: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>"
        ))
    
    add_trace(sorted_cs, accuracy_vals, 'Accuracy')
    add_trace(sorted_cs, precision_vals, 'Precision')
    add_trace(sorted_cs, recall_vals, 'Recall')
    add_trace(sorted_cs, f1_vals, 'F1-Score')

    # Style layout
    fig.update_layout(
        title="",
        xaxis_title="C Value",
        yaxis_title="Score",
        xaxis_type='log',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    return fig

# def grid_search_params_and_show_accuracy_heatmap(grid_search_results):




def node_link_topology_with_neuron_weights(model_path):
    """
    Visualize an sklearn MLPClassifier's topology and weights using Plotly,
    but with a fixed line width for all connections.
    """

    # 1. Load model and extract weights
    model = load(model_path)
    weights = [w.astype(float) for w in model.coefs_]

    # 2. Determine network architecture
    layers = [weights[0].shape[0]] + [w.shape[1] for w in weights[:-1]] + [weights[-1].shape[1]]
    num_layers = len(layers)

    # 3. Visualization parameters
    canvas_width = 1700
    canvas_height = 1300
    margin = 100
    neuron_radius = 16
    layer_spacing = (canvas_width // (num_layers - 1)) if num_layers > 1 else 300
    label_spacing = 20   # Vertical spacing between labels
    label_offset = 50    # Horizontal distance from neuron to label

    fig = go.Figure()

    def get_y_positions(num_nodes):
        """Calculate vertical positions for neurons."""
        if num_nodes == 1:
            return [float(canvas_height / 2)]
        return np.linspace(margin, canvas_height - margin, num_nodes).astype(float)

    # ========== LEGEND (Positive/Negative, etc.) ========== #
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=20, color="blue"),
        name="Positive Neuron Value",
        hoverinfo="none"
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=20, color="red"),
        name="Negative Neuron Value",
        hoverinfo="none"
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="lines",
        line=dict(color="gray", width=2),
        name="Positive Weight",
        hoverinfo="none"
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="lines",
        line=dict(color="red", width=2),
        name="Negative Weight",
        hoverinfo="none"
    ))

    # ========== NEURON NODES ========== #
    for layer_idx, num_nodes in enumerate(layers):
        x = layer_idx * layer_spacing
        y_positions = get_y_positions(num_nodes)

        for node_idx, y in enumerate(y_positions):
            # Calculate neuron value (sum of incoming weights)
            if layer_idx == 0:
                value = 0.0
            else:
                value = float(np.sum(weights[layer_idx - 1][:, node_idx]))

            color = "blue" if value >= 0 else "red"
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker=dict(size=neuron_radius * 2, color=color),
                text=[f"{value:.2f}"],
                textfont=dict(color="white", size=12),
                hoverinfo="text",
                hovertext=(
                    f"Layer {layer_idx + 1} Neuron {node_idx + 1}<br>"
                    f"Value: {value:.4f}"
                ),
                showlegend=False
            ))

    # ========== CONNECTIONS & VERTICAL LABELS ========== #
    connection_traces = []
    text_elements = []
    occupied_positions = {}

    def is_position_available(x, y):
        """Check if a position is available based on a grid of label_spacing."""
        key = (round(x / label_spacing), round(y / label_spacing))
        return key not in occupied_positions

    def mark_position_occupied(x, y):
        """Mark a position as occupied."""
        key = (round(x / label_spacing), round(y / label_spacing))
        occupied_positions[key] = True

    for layer_idx in range(len(weights)):
        w_matrix = weights[layer_idx]
        x_start = layer_idx * layer_spacing
        x_end = (layer_idx + 1) * layer_spacing

        source_y = get_y_positions(layers[layer_idx])
        target_y = get_y_positions(layers[layer_idx + 1])

        for src_idx in range(w_matrix.shape[0]):
            # Collect all weights for this source neuron
            weights_for_neuron = w_matrix[src_idx, :]
            base_x = x_start - label_offset
            base_y = source_y[src_idx]

            # Sort weights by absolute value (largest first)
            sorted_indices = np.argsort(-np.abs(weights_for_neuron))

            # Place labels vertically in that order
            for i, tgt_idx in enumerate(sorted_indices):
                weight = weights_for_neuron[tgt_idx]
                line_color = "gray" if weight >= 0 else "red"

                # Connection line coordinates
                y_start = source_y[src_idx]
                y_end = target_y[tgt_idx]

                # >>>>>> FIXED LINE WIDTH: width=2 <<<<<<
                connection_traces.append(go.Scatter(
                    x=[x_start, x_end],
                    y=[y_start, y_end],
                    mode="lines",
                    line=dict(
                        color=line_color,
                        width=2  # All lines have the same width now
                    ),
                    hoverinfo="text",
                    hovertext=(
                        f"From: Layer {layer_idx + 1} Neuron {src_idx + 1}<br>"
                        f"To: Layer {layer_idx + 2} Neuron {tgt_idx + 1}<br>"
                        f"Weight: {weight:.6f}"
                    ),
                    showlegend=False
                ))

                # Calculate label position
                label_x = base_x
                label_y = base_y + i * label_spacing

                # Ensure position is free
                while not is_position_available(label_x, label_y):
                    label_y += label_spacing  # Move down

                mark_position_occupied(label_x, label_y)

                # Add a small text marker (label)
                text_elements.append(go.Scatter(
                    x=[label_x],
                    y=[label_y],
                    mode="markers+text",
                    marker=dict(
                        size=10,
                        color="rgba(0,0,0,0)",
                        opacity=0
                    ),
                    text=[f"{weight:.2f}"],
                    textfont=dict(
                        color="black",
                        size=12,
                        family="Arial Bold"
                    ),
                    textposition="middle right",
                    hoverinfo="text",
                    hovertext=(
                        f"From: Layer {layer_idx + 1} Neuron {src_idx + 1}<br>"
                        f"To: Layer {layer_idx + 2} Neuron {tgt_idx + 1}<br>"
                        f"Weight: {weight:.6f}"
                    ),
                    showlegend=False
                ))

    # Add all traces in correct order
    fig.add_traces(connection_traces)  # Lines first
    fig.add_traces(text_elements)      # Then text

    # ========== LAYER LABELS ========== #
    for layer_idx, num_nodes in enumerate(layers):
        fig.add_annotation(
            x=layer_idx * layer_spacing,
            y=-150,
            text=f"Layer {layer_idx + 1}<br>({num_nodes} neurons)",
            showarrow=False,
            yanchor="top",
            font=dict(size=14)
        )

    # ========== FINAL LAYOUT ========== #
    fig.update_layout(
        title="Neural Network Topology (Fixed Connection Width)",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-100, canvas_width + 100]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-200, canvas_height + 100]
        ),
        height=canvas_height + 200,
        width=canvas_width + 200,
        margin=dict(b=150),
        plot_bgcolor="white",
        legend=dict(
            x=1.05,
            y=1,
            bordercolor="black",
            borderwidth=1,
            font=dict(size=12)
        ),
        hovermode="closest",
        hoverdistance=30
    )

    return fig


def convert_image_to_base64(image_path):
    """
    Convert an image file to base64 format for embedding in Dash.
    """
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"