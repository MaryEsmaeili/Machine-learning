import numpy as np

class BernoulliNBWithLaplace:
    def __init__(self, alpha=1.0):
        """
        Initialize the classifier with Laplace smoothing parameter alpha.
        
        Parameters:
        - alpha: Laplace smoothing parameter (default is 1.0).
        """
        self.alpha = alpha
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Train the Bernoulli Naive Bayes classifier with Laplace smoothing.
        
        Parameters:
        - X: Binary feature matrix (each feature should be 0 or 1).
        - y: Target labels (0 or 1 for binary classification).
        """
        # Identify unique classes and their counts
        self.classes_, class_counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Log prior probabilities for each class
        self.class_log_prior_ = np.log(class_counts / y.shape[0])

        # Initialize feature probabilities
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        # Calculate feature probabilities with Laplace smoothing
        for idx, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            # Laplace smoothing: add alpha to the numerator and 2*alpha to the denominator
            smoothed_counts = X_cls.sum(axis=0) + self.alpha
            smoothed_totals = X_cls.shape[0] + 2 * self.alpha
            self.feature_log_prob_[idx] = np.log(smoothed_counts / smoothed_totals)

    def predict_log_proba(self, X):
        """
        Calculate the log-probability estimates for each class.
        
        Parameters:
        - X: Feature matrix (each feature should be 0 or 1).
        
        Returns:
        - log_probs: Log-probability of each class for each sample.
        """
        # Calculate log-probabilities for each class and feature
        log_probs = []
        for idx, log_prior in enumerate(self.class_log_prior_):
            # Compute log-probability for class `idx`
            log_prob = log_prior + (X * self.feature_log_prob_[idx] + (1 - X) * (np.log(1 - np.exp(self.feature_log_prob_[idx])))).sum(axis=1)
            log_probs.append(log_prob)
        return np.array(log_probs).T

    def predict(self, X):
        """
        Predict the class labels for the given samples.
        
        Parameters:
        - X: Feature matrix (each feature should be 0 or 1).
        
        Returns:
        - Predicted class labels.
        """
        log_probs = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_probs, axis=1)]

# Usage example
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Load 20 Newsgroups data, restricting to two classes for binary classification
# categories = ['alt.atheism', 'soc.religion.christian']
# newsgroups = fetch_20newsgroups(categories=categories, subset='all')

# # Transform the text data into binary features
# vectorizer = CountVectorizer(binary=True)
# X = vectorizer.fit_transform(newsgroups.data).toarray()
# y = (newsgroups.target > 0).astype(int)  # Convert target to binary (0 or 1)

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Initialize and train the Bernoulli Naive Bayes classifier with Laplace smoothing
# model = BernoulliNBWithLaplace(alpha=1.0)
# model.fit(X_train, y_train)

# # Predict on the test set
# y_pred = model.predict(X_test)

# # Evaluate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy of Bernoulli Naive Bayes with Laplace Smoothing:", accuracy)
