import pandas as pd
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

    return X_train, X_test, y_train, y_test, X_train_pca, X_test_pca, pca