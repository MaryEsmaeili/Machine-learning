import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, lambda_=1.0):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_ = lambda_
        self.theta = None
        self.costs = []

    def sigmoid(self, z):
        """Compute the sigmoid function for a scalar or numpy array z."""
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y):
        """Compute the cost function with regularization for logistic regression."""
        m = len(y)
        h = self.sigmoid(X.dot(self.theta))
        regularization_term = (self.lambda_ / (2 * m)) * np.sum(self.theta[1:] ** 2)
        cost = (-1 / m) * (y.dot(np.log(h)) + (1 - y).dot(np.log(1 - h))) + regularization_term
        return cost

    def gradient_descent(self, X, y):
        """Perform gradient descent to learn theta."""
        m = len(y)
        self.costs = []

        for _ in range(self.num_iterations):
            h = self.sigmoid(X.dot(self.theta))
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

    def fit(self, X, y, scale=True):
        """Fit the logistic regression model to the data."""
        m, n = X.shape
        self.theta = np.zeros(n + 1)  # Include theta_0
        X = np.c_[np.ones(m), X]  # Add intercept term

        # Perform gradient descent
        self.theta, self.costs = self.gradient_descent(X, y)

    def predict_proba(self, X):
        """Predict probabilities using the logistic regression model."""
        m = X.shape[0]
        X = np.c_[np.ones(m), X]  # Add intercept
        return self.sigmoid(X.dot(self.theta))

    def predict(self, X, threshold=0.5):
        """Predict binary class labels (0 or 1) based on the probability threshold."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def plot_cost(self):
        """Plot the cost function over iterations to visualize convergence."""
        import matplotlib.pyplot as plt
        plt.plot(range(len(self.costs)), self.costs, color="blue")
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Cost Function Convergence")
        plt.show()

# Usage example
# X = data[['feature1', 'feature2', 'feature3']].values  # replace with actual feature names
# y = data['target'].values  # replace with the actual target name

# # Initialize and train the logistic regression model
# model = LogisticRegression(learning_rate=0.01, num_iterations=1000, lambda_=1.0)
# model.fit(X, y)

# # Plot cost function to verify convergence
# model.plot_cost()

# # Make predictions on new data
# predictions = model.predict(X)  # Class labels
# probabilities = model.predict_proba(X)  # Probabilities
