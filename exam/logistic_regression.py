import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iters=10000, lambda_=0.1):
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.lambda_ = lambda_
        self.theta = None
        self.cost_history = []
        
    def sigmoid(self, z):
        """
        Compute the sigmoid of z, with handling for large values.
        """
        z = np.clip(z, -500, 500)  # Prevent overflow in exp
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, X, y, theta):
        """
        Compute the cost function for logistic regression with regularization.
        """
        m = len(y)
        h = self.sigmoid(X @ theta)

        # Clip predicted values to avoid log(0) error
        h = np.clip(h, 1e-10, 1 - 1e-10)

        # Compute the cost with regularization (excluding theta[0])
        reg_term = (self.lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
        cost = -1 / m * (y @ np.log(h) + (1 - y) @ np.log(1 - h)) + reg_term
        
        return cost
    
    def gradient_descent(self, X, y):
        """
        Perform gradient descent to learn theta.
        """
        m = len(y)
        self.theta = np.zeros(X.shape[1])
        self.cost_history = []

        for i in range(self.num_iters):
            h = self.sigmoid(X @ self.theta)
            error = h - y

            # Update theta (for j >= 1, apply regularization)
            self.theta[0] -= self.learning_rate * (1 / m) * (X[:, 0].T @ error)
            self.theta[1:] -= self.learning_rate * ((1 / m) * (X[:, 1:].T @ error) + (self.lambda_ / m) * self.theta[1:])
            
            # Save the cost in history
            cost = self.compute_cost(X, y, self.theta)
            self.cost_history.append(cost)
        
        return self.theta
    
    def fit(self, X, y):
        """
        Train the logistic regression model.
        """
        X = np.c_[np.ones(X.shape[0]), X]  # Add intercept term
        self.theta = self.gradient_descent(X, y)
    
    def predict_proba(self, X):
        """
        Predict probabilities for each class.
        """
        X = np.c_[np.ones(X.shape[0]), X]  # Add intercept term
        return self.sigmoid(X @ self.theta)
    
    def predict(self, X):
        """
        Predict the class labels (0 or 1).
        """
        return self.predict_proba(X) >= 0.5
