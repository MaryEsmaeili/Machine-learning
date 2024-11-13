# gaussian_nb.py
import numpy as np

class GaussianNBClassifier:
    def __init__(self):
        self.labels = None
        self.mu = None
        self.sd = None
        self.priors = None

    def fit(self, X, y):
        self.labels = np.unique(y)
        n_classes = len(self.labels)
        n_features = X.shape[1]

        self.mu = np.zeros((n_classes, n_features))
        self.sd = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        for idx, label in enumerate(self.labels):
            X_class = X[y == label]
            self.mu[idx, :] = X_class.mean(axis=0)
            self.sd[idx, :] = X_class.std(axis=0)
            self.priors[idx] = X_class.shape[0] / X.shape[0]

    def predict(self, X):
        n_samples = X.shape[0]
        posteriors = np.zeros((n_samples, len(self.labels)))

        for idx, label in enumerate(self.labels):
            likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.sd[idx, :] ** 2))
            exponent = -0.5 * np.sum(((X - self.mu[idx, :]) ** 2) / (self.sd[idx, :] ** 2), axis=1)
            log_prior = np.log(self.priors[idx])
            posteriors[:, idx] = likelihood + exponent + log_prior

        return self.labels[np.argmax(posteriors, axis=1)]
