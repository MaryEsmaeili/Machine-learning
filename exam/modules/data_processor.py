# dataprocessor.py

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import numpy as np

class DataProcessor:
    def __init__(self, k_best_features=10, apply_log=False, apply_scaling=True, apply_pca=False, pca_components=2):
        self.k_best_features = k_best_features
        self.apply_log = apply_log
        self.apply_scaling = apply_scaling
        self.apply_pca = apply_pca
        self.pca_components = pca_components

        self.selector = None
        self.scaler = StandardScaler() if apply_scaling else None
        self.pca = PCA(n_components=pca_components) if apply_pca else None

    def fit_transform(self, X_train, y_train):
        X_train = self.create_features(X_train)  # Generate new features
        if self.apply_log:
            X_train = self.log_transform(X_train)  # Apply log transform

        X_train = self.encode_features(X_train)  # Encode categorical features

        # Feature selection
        self.selector = SelectKBest(score_func=f_classif, k=min(self.k_best_features, X_train.shape[1]))
        X_train_selected = self.selector.fit_transform(X_train, y_train)

        # Scaling
        if self.apply_scaling:
            X_train_selected = self.scaler.fit_transform(X_train_selected)

        # Dimensionality reduction
        if self.apply_pca:
            X_train_selected = self.pca.fit_transform(X_train_selected)

        return X_train_selected

    def transform(self, X_test):
        X_test = self.create_features(X_test)
        if self.apply_log:
            X_test = self.log_transform(X_test)
        
        X_test = self.encode_features(X_test)

        X_test_selected = self.selector.transform(X_test) if self.selector else X_test
        if self.apply_scaling:
            X_test_selected = self.scaler.transform(X_test_selected)
        if self.apply_pca:
            X_test_selected = self.pca.transform(X_test_selected)

        return X_test_selected

    def create_features(self, X):
        """Add squared terms for numeric columns at once to avoid fragmentation."""
        squared_features = {f'{col}_squared': X[col] ** 2 for col in X.select_dtypes(include=[float, int]).columns}
        return pd.concat([X, pd.DataFrame(squared_features, index=X.index)], axis=1)

    def log_transform(self, X):
        for col in X.select_dtypes(include=[float, int]).columns:
            if (X[col] > 0).all():
                X[f'log_{col}'] = np.log(X[col])
        return X

    def encode_features(self, X):
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            encoder = OneHotEncoder(sparse=False, drop='first')
            X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]), index=X.index)
            X = X.drop(columns=categorical_cols).join(X_encoded)
        return X
