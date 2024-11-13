import numpy as np

class PolynomialRegression:
    def __init__(self, degree=2, learning_rate=0.01, num_iters=10000):
        """
        Polynomial Regression model.

        Parameters:
        degree (int): Degree of the polynomial.
        learning_rate (float): Step size for gradient descent.
        num_iters (int): Number of iterations for gradient descent.
        """
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.theta = None

    def polynomial_features(self, X):
        """
        Expands the input data into polynomial features.

        Parameters:
        X (ndarray): Feature matrix.

        Returns:
        ndarray: Polynomial features matrix.
        """
        X_poly = X.copy()
        for i in range(2, self.degree + 1):
            X_poly = np.c_[X_poly, X**i]  # Add polynomial terms
        
        return X_poly

    def fit(self, X, y):
        """
        Fit the polynomial regression model using gradient descent.

        Parameters:
        X (ndarray): Feature matrix.
        y (ndarray): Target vector.
        """
        # Expand features to include polynomial terms
        X_poly = self.polynomial_features(X)
        
        # Add intercept term (bias)
        X_poly = np.c_[np.ones(X_poly.shape[0]), X_poly]

        # Initialize theta (parameters)
        self.theta = np.zeros(X_poly.shape[1])

        m = len(y)  # Number of training examples
        for _ in range(self.num_iters):
            h = X_poly @ self.theta  # Linear hypothesis
            error = h - y
            gradient = (1 / m) * (X_poly.T @ error)
            self.theta -= self.learning_rate * gradient  # Update theta
    
    def predict(self, X):
        """
        Predict the target values using the trained polynomial regression model.

        Parameters:
        X (ndarray): Feature matrix.

        Returns:
        ndarray: Predicted values.
        """
        # Expand features to include polynomial terms
        X_poly = self.polynomial_features(X)
        
        # Add intercept term (bias)
        X_poly = np.c_[np.ones(X_poly.shape[0]), X_poly]

        return X_poly @ self.theta

    def get_params(self):
        """
        Get the learned parameters (theta).
        """
        return self.theta
