import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import precision_score, recall_score, f1_score


# 1. prepare the data (load, clean, split, standardize, pca)
# Load the dataset
data = pd.read_csv('pulsar_data.csv')

# Handle missing values by replacing them with the mean of the column
data_cleaned = data.fillna(data.mean())

# Feature X and target y
X = data_cleaned.drop(columns=["target_class"])
y = data_cleaned["target_class"]

# Split the data into training and test data (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reduce data to 2D for visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


