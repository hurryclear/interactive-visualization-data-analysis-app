import os
import json
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from pre_data import pre_data
from helper_visualization import calculate_accuracy, learning_curves_dff, confusion_matrix_dff, block_topology, node_link_topology_with_neuron_weights

# File paths for saved artifacts
MODEL1_PATH = "./model1/dff_model.h5"
MODEL1_HISTORY_PATH = "./model1/dff_training_history.json"
MODEL1_X_TEST_PATH = "./model1/dff_X_test.npy"
MODEL1_Y_TEST_PATH = "./model1/dff_y_test.npy"
MODEL1_EVAL_PATH = "./model1/dff_evaluation_metrics.json"

MODEL2_PATH = "./model2/dff_model.h5"
MODEL2_HISTORY_PATH = "./model2/dff_training_history.json"
MODEL2_X_TEST_PATH = "./model2/dff_X_test.npy"
MODEL2_Y_TEST_PATH = "./model2/dff_y_test.npy"
MODEL2_EVAL_PATH = "./model2/dff_evaluation_metrics.json"

# 1. Prepare the data
X_train, X_test, y_train, y_test, X_train_pca, X_test_pca, pca = pre_data(2)

def train_and_save_model1():

    # 2. Build the network
    model = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 3. Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # 4. Evaluate the model
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Save evaluation metrics
    evaluation = {
        "confusion_matrix": conf_matrix.tolist(),  # Convert numpy array to list for JSON serialization
        "classification_report": class_report
    }
    with open(MODEL1_EVAL_PATH, 'w') as f:
        json.dump(evaluation, f)

    # 4. Save the model and related data
    model.save(MODEL1_PATH)
    with open(MODEL1_HISTORY_PATH, 'w') as f:
        json.dump(history.history, f)
    np.save(MODEL1_X_TEST_PATH, X_test)
    np.save(MODEL1_Y_TEST_PATH, y_test)

    print(f"Model, history, and evaluation metrics saved: {MODEL1_PATH}, {MODEL1_HISTORY_PATH}, {MODEL1_EVAL_PATH}")


def train_and_save_model2():
    # 1. Build the shallow neural network
    model = Sequential([
        Dense(32, activation='relu', input_dim=X_train.shape[1]),  # Single hidden layer with 32 neurons
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 2. Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,  # 20% of the training data used for validation
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # 3. Evaluate the model
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Save evaluation metrics
    evaluation = {
        "confusion_matrix": conf_matrix.tolist(),  # Convert numpy array to list for JSON serialization
        "classification_report": class_report
    }
    with open(MODEL2_EVAL_PATH, 'w') as f:
        json.dump(evaluation, f)

    # Save the model and related data
    model.save(MODEL2_PATH)
    with open(MODEL2_HISTORY_PATH, 'w') as f:
        json.dump(history.history, f)
    np.save(MODEL2_X_TEST_PATH, X_test)
    np.save(MODEL2_Y_TEST_PATH, y_test)

    print(f"Shallow model, history, and evaluation metrics saved: {MODEL2_PATH}, {MODEL2_HISTORY_PATH}, {MODEL2_EVAL_PATH}")



# # Helper functions for visualization
# def calculate_accuracy(confusion_matrix):
#     """
#     Calculate accuracy from the confusion matrix.
#     """
#     true_positives_and_negatives = sum(confusion_matrix[i][i] for i in range(len(confusion_matrix)))
#     total_samples = sum(sum(row) for row in confusion_matrix)
#     accuracy = true_positives_and_negatives / total_samples
#     return accuracy

# def learning_curves_dff(history):
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         y=history['accuracy'], mode='lines+markers', name='Train Accuracy'))
#     fig.add_trace(go.Scatter(
#         y=history['val_accuracy'], mode='lines+markers', name='Validation Accuracy'))
#     fig.add_trace(go.Scatter(
#         y=history['loss'], mode='lines+markers', name='Train Loss'))
#     fig.add_trace(go.Scatter(
#         y=history['val_loss'], mode='lines+markers', name='Validation Loss'))
#     fig.update_layout(
#         title="Learning Curves",
#         xaxis_title="Epochs",
#         yaxis_title="Metrics",
#         legend_title="Metrics"
#     )
#     return fig

# def confusion_matrix_dff(conf_matrix):
#     fig = go.Figure(data=go.Heatmap(
#         z=conf_matrix,
#         x=["Predicted 0", "Predicted 1"],
#         y=["Actual 0", "Actual 1"],
#         colorscale="Blues",
#         showscale=True,
#         text=conf_matrix,
#         texttemplate="%{text}"
#     ))
#     fig.update_layout(
#         title="Confusion Matrix",
#         xaxis_title="Predicted",
#         yaxis_title="Actual"
#     )
#     return fig

# def block_topology(model_path, output_file):
#     """
#     Visualize the topology of the neural network and save as an image.
#     """
#     model = load_model(model_path)
#     plot_model(
#         model,
#         to_file=output_file,
#         show_shapes=True,
#         show_layer_names=True,
#         dpi=96
#     )
#     return output_file

# def node_link_topology_with_neuron_weights(model_path):
#     fig = go.Figure()

#     # Load the model
#     model = load_model(model_path)

#     # Get input neurons (from model's input tensor shape)
#     input_neurons = model.input_shape[-1]  # Input shape's last dimension (features)

#     # Get hidden and output layers
#     hidden_layers = []
#     weights = []  # To store weights for connections
#     for layer in model.layers:
#         if hasattr(layer, "units"):  # Only Dense layers have the 'units' attribute
#             hidden_layers.append(layer.units)
#         if hasattr(layer, "get_weights"):  # Get weights for Dense layers
#             layer_weights = layer.get_weights()
#             if len(layer_weights) > 0:  # First item is the weight matrix
#                 weights.append(layer_weights[0])

#     # Output neurons (last layer in the hidden_layers list)
#     output_neurons = hidden_layers.pop()

#     # Define the layers (input, hidden, output)
#     layers = [input_neurons] + hidden_layers + [output_neurons]
#     num_layers = len(layers)
#     canvas_width = 1800  # Fixed canvas width
#     canvas_height = 2500  # Fixed canvas height
#     neuron_radius = 16  # Radius of the neurons for display
#     layer_spacing = canvas_width // (num_layers - 1)  # Dynamically calculate horizontal spacing
#     # Add nodes for each layer
#     neuron_values = {}  # Store values (e.g., sum of incoming weights) for each neuron
#     for layer_idx, num_nodes in enumerate(layers):
#         x_position = layer_idx * layer_spacing
#         # Calculate vertical offset to center neurons in the canvas
#         y_offset = (canvas_height - num_nodes * 50) / 2
#         for node_idx in range(num_nodes):
#             y_position = y_offset + node_idx * 50

#             if layer_idx == 0:  # Input layer neurons
#                 neuron_value = 1.0  # Placeholder value for input neurons
#             else:
#                 # Compute the sum of weights coming into this neuron
#                 neuron_value = np.sum(weights[layer_idx - 1][:, node_idx])

#             # Store neuron value
#             neuron_values[(layer_idx, node_idx)] = neuron_value

#             # Add neuron node
#             fig.add_trace(go.Scatter(
#                 x=[x_position],
#                 y=[y_position],
#                 mode="markers+text",
#                 marker=dict(
#                     size=neuron_radius * 2,
#                     color="blue" if neuron_value >= 0 else "red",  # Blue for positive, red for negative
#                 ),
#                 text=[f"{neuron_value:.2f}"],  # Show neuron value
#                 textposition="middle center",
#                 textfont=dict(
#                     size=12,        # Font size
#                     color="white"   # Text color
#                 ),
#                 hoverinfo="text",
#                 name=f"Neuron {node_idx + 1} in Layer {layer_idx + 1}"
#             ))

#     # Add connections between layers
#     for layer_idx, weight_matrix in enumerate(weights):
#         x_start = layer_idx * layer_spacing
#         x_end = (layer_idx + 1) * layer_spacing
#         y_start_offset = (canvas_height - layers[layer_idx] * 50) / 2
#         y_end_offset = (canvas_height - layers[layer_idx + 1] * 50) / 2
#         for source_idx in range(weight_matrix.shape[0]):  # For each neuron in the current layer
#             y_start = y_start_offset + source_idx * 50
#             for target_idx in range(weight_matrix.shape[1]):  # For each neuron in the next layer
#                 y_end = y_end_offset + target_idx * 50
#                 weight = weight_matrix[source_idx, target_idx]
#                 fig.add_trace(go.Scatter(
#                     x=[x_start, x_end],
#                     y=[y_start, y_end],
#                     mode="lines",
#                     line=dict(
#                         color="gray" if weight > 0 else "red",  # Gray for positive, red for negative weights
#                         width=np.abs(weight) * 3  # Thickness proportional to the absolute weight value
#                     ),
#                     hoverinfo="text",
#                     text=[f"Weight: {weight:.4f}"],
#                     showlegend=False
#                 ))

#     # Add layer labels
#     for layer_idx, num_nodes in enumerate(layers):
#         x_position = layer_idx * layer_spacing
#         fig.add_trace(go.Scatter(
#             x=[x_position],
#             y=[-500],  # Position labels below the nodes
#             mode="text",
#             text=[f"Layer {layer_idx + 1} ({num_nodes} Neurons)"],
#             textfont=dict(
#                 size=20,        # Font size
#                 color="black"   # Text color
#             ),
#             textposition="bottom center",
#             showlegend=False
#         ))

#     # Adjust layout
#     fig.update_layout(
#         title="Neural Network Topology with Vertically Centered Neuron Weights (Full Width)",
#         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#         showlegend=False,
#         height=canvas_height + 100,
#         width=canvas_width + 200  # Add padding for better visualization
#     )
#     return fig

if __name__ == "__main__":
    train_and_save_model1()
    train_and_save_model2()