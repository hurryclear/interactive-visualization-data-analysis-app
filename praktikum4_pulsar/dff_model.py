import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from helper_functions import pre_data

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
# X_train, X_test, y_train, y_test, X_train_pca, X_test_pca, pca = pre_data(2)

data1 = pre_data(2)
data2 = pre_data(2)

def train_and_save_and_evaluate_model1(X_train, X_test, y_train, y_test):

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


def train_and_save_and_evaluate_model2(X_train, X_test, y_train, y_test):
    # 1. Build the shallow neural network
    model = Sequential([
        Dense(32, activation='relu', input_dim=X_train.shape[1]),  # Single hidden layer with 32 neurons
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 2. Train the model
    # early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,  # 20% of the training data used for validation
        epochs=50,
        batch_size=32,
        # callbacks=[early_stopping],
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



if __name__ == "__main__":
    train_and_save_and_evaluate_model1(data1[0], data1[1], data1[2], data1[3])
    train_and_save_and_evaluate_model2(data2[0], data2[1], data2[2], data2[3])