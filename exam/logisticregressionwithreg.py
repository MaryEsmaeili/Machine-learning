import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionWithReg:
    def __init__(self, learning_rate=0.01, num_iterations=1000, lambda_=1.0):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_ = lambda_
        self.theta = None
        self.costs = []

    def data_preparation(self, data, target_column):
        """Prepare feature matrix X, target vector y, and initialize theta."""
        # Extract target vector y
        y = data[target_column].values  # Convert to numpy array
        
        # Extract feature matrix X by dropping the target column
        X = data.drop(columns=[target_column]).values  # Convert to numpy array
        
        # Add a column of ones to X for theta_0
        X = np.c_[np.ones(X.shape[0]), X]
        
        # Initialize theta vector with zeros
        self.theta = np.zeros(X.shape[1])
        
        return X, y

    def compute_cost(self, X, y):
        """Compute the cost with regularization."""
        m = len(y)
        h = X.dot(self.theta)
        cost = (1 / (2 * m)) * np.sum((h - y) ** 2)
        regularization_term = (self.lambda_ / (2 * m)) * np.sum(self.theta[1:] ** 2)
        return cost + regularization_term

    def gradient_descent(self, X, y):
        """Perform gradient descent to learn theta."""
        m = len(y)
        self.costs = []
        
        for _ in range(self.num_iterations):
            h = X.dot(self.theta)
            error = h - y
            
            # Gradient calculation with regularization (except for theta_0)
            gradient = (1 / m) * X.T.dot(error)
            gradient[1:] += (self.lambda_ / m) * self.theta[1:]
            
            # Update theta
            self.theta -= self.learning_rate * gradient
            
            # Record cost for this iteration
            cost = self.compute_cost(X, y)
            self.costs.append(cost)

        return self.theta, self.costs

    def fit(self, data, target_column, scale=True):
        """Prepare data, scale if needed, and train the model."""
        X, y = self.data_preparation(data, target_column)
        
        # Scale X if necessary
        if scale:
            self.scaler_mean = X[:, 1:].mean(axis=0)
            self.scaler_std = X[:, 1:].std(axis=0)
            X[:, 1:] = (X[:, 1:] - self.scaler_mean) / self.scaler_std
        
        # Perform gradient descent
        self.theta, self.costs = self.gradient_descent(X, y)
        
        # Adjust theta back to original scale if necessary
        if scale:
            self.theta = self.reverse_theta()

    def reverse_theta(self):
        """Transform theta back to original scale if X was standardized."""
        theta_original = np.zeros_like(self.theta)
        theta_original[1:] = self.theta[1:] / self.scaler_std
        theta_original[0] = self.theta[0] - np.sum((self.theta[1:] * self.scaler_mean) / self.scaler_std)
        return theta_original

    def predict(self, X):
        """Make predictions using the trained model."""
        X = np.c_[np.ones(X.shape[0]), X]  # Add intercept
        return X.dot(self.theta)

    def plot_cost(self):
        """Plot the cost function over iterations."""
        plt.plot(range(len(self.costs)), self.costs, color="blue")
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Cost Function Convergence")
        plt.show()

# Example usage:
# Assuming `data` is a pandas DataFrame with features and a target column
# model = LogisticRegressionWithReg(learning_rate=0.01, num_iterations=1000, lambda_=1.0)
# model.fit(data, target_column="target", scale=True)
# model.plot_cost()
# predictions = model.predict(X_new)  # where X_new is the new feature matrix
