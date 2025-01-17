import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# File paths for saved artifacts
MODEL_PATH = "dff_model.h5"
HISTORY_PATH = "dff_training_history.json"
X_TEST_PATH = "dff_X_test.npy"
Y_TEST_PATH = "dff_y_test.npy"

def train_and_save_model():
    # 1. Prepare the data
    data = pd.read_csv('pulsar_data.csv')
    data_cleaned = data.fillna(data.mean())
    X = data_cleaned.drop(columns=["target_class"])
    y = data_cleaned["target_class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Optional: Reduce data to 2D for visualization using PCA
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

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

    # 4. Save the model and related data
    model.save(MODEL_PATH)
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history.history, f)
    np.save(X_TEST_PATH, X_test)
    np.save(Y_TEST_PATH, y_test)

    print(f"Model and related data saved: {MODEL_PATH}, {HISTORY_PATH}, {X_TEST_PATH}, {Y_TEST_PATH}")

if __name__ == "__main__":
    train_and_save_model()