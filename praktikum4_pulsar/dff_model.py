import os
import json
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from pre_data import pre_data

# File paths for saved artifacts
MODEL_PATH = "dff_model.h5"
HISTORY_PATH = "dff_training_history.json"
X_TEST_PATH = "dff_X_test.npy"
Y_TEST_PATH = "dff_y_test.npy"
EVAL_PATH = "dff_evaluation_metrics.json"

def train_and_save_model():
    # 1. Prepare the data
    X_train, X_test, y_train, y_test, X_train_pca, X_test_pca, pca = pre_data(2)

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
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save evaluation metrics
    evaluation = {
        "confusion_matrix": conf_matrix.tolist(),  # Convert numpy array to list for JSON serialization
        "classification_report": report
    }
    with open(EVAL_PATH, 'w') as f:
        json.dump(evaluation, f)

    # 4. Save the model and related data
    model.save(MODEL_PATH)
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history.history, f)
    np.save(X_TEST_PATH, X_test)
    np.save(Y_TEST_PATH, y_test)

    print(f"Model, history, and evaluation metrics saved: {MODEL_PATH}, {HISTORY_PATH}, {EVAL_PATH}")


# Helper functions for visualization

def create_learning_curves(history):
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

def create_confusion_matrix(conf_matrix):
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



if __name__ == "__main__":
    train_and_save_model()