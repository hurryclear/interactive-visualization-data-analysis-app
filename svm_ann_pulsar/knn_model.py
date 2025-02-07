import json
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
from helper_functions import pre_data  # Assuming you still have a function to load/prepare data

# File paths for saved artifacts (using joblib instead of .h5)
MODEL1_PATH = "./model1/knn_model.joblib"
MODEL1_HISTORY_PATH = "./model1/knn_training_history.json"
MODEL1_X_TEST_PATH = "./model1/knn_X_test.npy"
MODEL1_Y_TEST_PATH = "./model1/knn_y_test.npy"
MODEL1_EVAL_PATH = "./model1/knn_evaluation_metrics.json"

MODEL2_PATH = "./model2/knn_model.joblib"
MODEL2_HISTORY_PATH = "./model2/knn_training_history.json"
MODEL2_X_TEST_PATH = "./model2/knn_X_test.npy"
MODEL2_Y_TEST_PATH = "./model2/knn_y_test.npy"
MODEL2_EVAL_PATH = "./model2/knn_evaluation_metrics.json"

# Example data retrieval
# data1 = pre_data(2) -> returns (X_train, X_test, y_train, y_test, ...)
# data2 = pre_data(2)

data1 = pre_data(2)
data2 = pre_data(2)

def train_and_save_and_evaluate_model1(X_train, X_test, y_train, y_test):

    # 1. Build the classifier
    #   'relu' is used for hidden layers. The final output layer is logistic for binary classification
    model = MLPClassifier(
        hidden_layer_sizes=(),
        activation='logistic',
        solver='adam',
        alpha=0.0001,
        max_iter=100,             # You can adjust epochs here
        early_stopping=True,       # Mimics early stopping
        n_iter_no_change=5,        # Patience
        validation_fraction=0.2,   # Equivalent to validation_split=0.2
        random_state=42,
        verbose=True
    )

    # 2. Train the model
    model.fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs > 0.5).astype(int)

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    evaluation = {
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": class_report
    }

    # 4. Save evaluation metrics
    with open(MODEL1_EVAL_PATH, 'w') as f:
        json.dump(evaluation, f)

    # 5. Save the model
    dump(model, MODEL1_PATH)

    # 6. Save "training history"
    # MLPClassifier provides:
    #   - loss_curve_ (list of loss values at each iteration)
    #   - validation_scores_ (list of scores on validation set at each iteration if early_stopping=True)
    history = {
        "loss_curve": model.loss_curve_,
    }
    # Only present if early_stopping=True
    if hasattr(model, "validation_scores_"):
        history["validation_scores"] = model.validation_scores_

    with open(MODEL1_HISTORY_PATH, 'w') as f:
        json.dump(history, f)

    # 7. Save test data for future reference
    np.save(MODEL1_X_TEST_PATH, X_test)
    np.save(MODEL1_Y_TEST_PATH, y_test)

    print(f"Model, history, and evaluation metrics saved: {MODEL1_PATH}, {MODEL1_HISTORY_PATH}, {MODEL1_EVAL_PATH}")


def train_and_save_and_evaluate_model2(X_train, X_test, y_train, y_test):

    # 1. Build the shallow MLPClassifier
    model = MLPClassifier(
        hidden_layer_sizes=(8,),  # single hidden layer of size 32
        activation='relu',
        solver='adam',
        alpha=0.0001,
        max_iter=100,
        early_stopping=True,
        n_iter_no_change=5,
        validation_fraction=0.2,
        random_state=42,
        verbose=True
    )

    # 2. Train the model
    model.fit(X_train, y_train)

    # 3. Evaluate the model
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs > 0.5).astype(int)

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    evaluation = {
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": class_report
    }

    with open(MODEL2_EVAL_PATH, 'w') as f:
        json.dump(evaluation, f)

    # 4. Save the model
    dump(model, MODEL2_PATH)

    # 5. Save "training history"
    history = {
        "loss_curve": model.loss_curve_,
    }
    if hasattr(model, "validation_scores_"):
        history["validation_scores"] = model.validation_scores_

    with open(MODEL2_HISTORY_PATH, 'w') as f:
        json.dump(history, f)

    # 6. Save test data
    np.save(MODEL2_X_TEST_PATH, X_test)
    np.save(MODEL2_Y_TEST_PATH, y_test)

    print(f"Shallow model, history, and evaluation metrics saved: {MODEL2_PATH}, {MODEL2_HISTORY_PATH}, {MODEL2_EVAL_PATH}")


if __name__ == "__main__":
    # Extract the relevant parts from the data returned by pre_data
    X_train1, X_test1, y_train1, y_test1 = data1[0], data1[1], data1[2], data1[3]
    X_train2, X_test2, y_train2, y_test2 = data2[0], data2[1], data2[2], data2[3]

    # Train, save, and evaluate the deeper MLP
    train_and_save_and_evaluate_model1(X_train1, X_test1, y_train1, y_test1)
    # Train, save, and evaluate the shallow MLP
    train_and_save_and_evaluate_model2(X_train2, X_test2, y_train2, y_test2)