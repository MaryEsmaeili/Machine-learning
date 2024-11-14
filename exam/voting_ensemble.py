from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import clone
import numpy as np

class VotingEnsemble:
    def __init__(self, models, voting_type='hard'):
        """
        Initialize the VotingEnsemble with a list of models and voting type.
        
        Parameters:
        - models: List of (name, model) tuples where each model is a classifier.
        - voting_type: 'hard' for majority voting, 'soft' for averaging predicted probabilities.
        """
        self.models = models
        self.voting_type = voting_type
        self.fitted_models = []

    def fit(self, X, y):
        """
        Fit all models in the ensemble to the training data.
        
        Parameters:
        - X: Feature matrix for training.
        - y: Target labels for training.
        """
        self.fitted_models = [(name, clone(model).fit(X, y)) for name, model in self.models]

    def predict(self, X):
        """
        Make predictions based on the voting type.
        
        Parameters:
        - X: Feature matrix for prediction.
        
        Returns:
        - Final ensemble predictions based on voting type.
        """
        if self.voting_type == 'hard':
            # Hard voting (majority vote)
            predictions = np.array([model.predict(X) for _, model in self.fitted_models])
            final_prediction = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        elif self.voting_type == 'soft':
            # Soft voting (average of probabilities)
            probabilities = np.array([model.predict_proba(X) for _, model in self.fitted_models])
            avg_probabilities = np.mean(probabilities, axis=0)
            final_prediction = np.argmax(avg_probabilities, axis=1)
        else:
            raise ValueError("Invalid voting_type. Choose 'hard' or 'soft'.")
        
        return final_prediction

    def evaluate(self, X, y):
        """
        Evaluate the ensemble model on test data and print accuracy.
        
        Parameters:
        - X: Feature matrix for evaluation.
        - y: True labels for evaluation.
        
        Returns:
        - Accuracy of the ensemble model.
        """
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"Ensemble {self.voting_type.capitalize()} Voting Accuracy: {accuracy}")
        return accuracy

    def compare_with_bagging_boosting(self, X_train, y_train, X_test, y_test):
        """
        Compare the performance of the voting ensemble with bagging and boosting algorithms.
        
        Parameters:
        - X_train, y_train: Training data for fitting models.
        - X_test, y_test: Test data for evaluation.
        
        Returns:
        - Dictionary containing accuracy scores of the ensemble, bagging, and boosting models.
        """
        results = {}
        
        # Evaluate voting ensemble
        print("Evaluating Voting Ensemble...")
        self.fit(X_train, y_train)
        results['VotingEnsemble'] = self.evaluate(X_test, y_test)
        
        # Evaluate Random Forest (bagging example)
        print("Evaluating Random Forest (Bagging)...")
        bagging_model = RandomForestClassifier(random_state=42)
        bagging_model.fit(X_train, y_train)
        bagging_accuracy = accuracy_score(y_test, bagging_model.predict(X_test))
        print(f"Bagging (Random Forest) Accuracy: {bagging_accuracy}")
        results['Bagging (Random Forest)'] = bagging_accuracy
        
        # Evaluate Gradient Boosting (boosting example)
        print("Evaluating Gradient Boosting...")
        boosting_model = GradientBoostingClassifier(random_state=42)
        boosting_model.fit(X_train, y_train)
        boosting_accuracy = accuracy_score(y_test, boosting_model.predict(X_test))
        print(f"Boosting (Gradient Boosting) Accuracy: {boosting_accuracy}")
        results['Boosting (Gradient Boosting)'] = boosting_accuracy
        
        return results
