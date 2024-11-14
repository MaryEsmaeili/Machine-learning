from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class ClassificationPipeline:
    def __init__(self, model_type='RandomForest', param_grid=None):
        """
        Initialize the ClassificationPipeline with a chosen model and parameter grid.
        
        Parameters:
        - model_type: Type of classifier ('RandomForest' or 'SVM').
        - param_grid: Dictionary containing parameter grid for GridSearchCV.
        """
        self.model_type = model_type
        self.param_grid = param_grid if param_grid else self.default_param_grid()
        self.pipeline = None
        self.grid_search = None

    def default_param_grid(self):
        """
        Provide a default parameter grid based on the model type.
        """
        if self.model_type == 'RandomForest':
            return {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10]
            }
        elif self.model_type == 'SVM':
            return {
                'model__C': [0.1, 1, 10, 100],
                'model__gamma': [1, 0.1, 0.01, 0.001],
                'model__kernel': ['rbf', 'linear']
            }
        else:
            raise ValueError("Unsupported model type. Choose 'RandomForest' or 'SVM'.")

    def create_pipeline(self):
        """
        Set up the pipeline with data preprocessing and the classifier.
        """
        # Choose model based on model type
        if self.model_type == 'RandomForest':
            model = RandomForestClassifier(random_state=42)
        elif self.model_type == 'SVM':
            model = SVC(probability=True, random_state=42)
        else:
            raise ValueError("Unsupported model type. Choose 'RandomForest' or 'SVM'.")
        
        # Define the pipeline steps
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Data Preprocessing: Standard Scaling
            ('model', model)               # Classifier
        ])

    def perform_grid_search(self, X_train, y_train, cv=5):
        """
        Perform GridSearchCV on the pipeline to find the best hyperparameters.
        
        Parameters:
        - X_train: Feature matrix for training.
        - y_train: Target labels for training.
        - cv: Cross-validation strategy (default is 5-fold).
        
        Returns:
        - Best parameters found through GridSearchCV.
        """
        self.create_pipeline()
        # Initialize GridSearchCV with the pipeline and parameter grid
        self.grid_search = GridSearchCV(estimator=self.pipeline, param_grid=self.param_grid, cv=cv, scoring='accuracy')
        
        # Fit the grid search to the training data
        self.grid_search.fit(X_train, y_train)
        
        print("Best Parameters:", self.grid_search.best_params_)
        return self.grid_search.best_params_

    def evaluate(self, X_test, y_test):
        """
        Evaluate the pipeline with the best parameters on test data.
        
        Parameters:
        - X_test: Feature matrix for testing.
        - y_test: True labels for testing.
        
        Returns:
        - Accuracy score on test data.
        """
        if self.grid_search is None:
            raise Exception("Grid search not performed. Call perform_grid_search() first.")
        
        # Use the best estimator found by GridSearchCV to predict on test data
        y_pred = self.grid_search.best_estimator_.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Test Accuracy:", accuracy)
        return accuracy

# Usage example
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split

# # Generate a sample dataset
# X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Initialize the pipeline with RandomForest and a parameter grid
# pipeline = ClassificationPipeline(model_type='RandomForest')
# # Perform grid search to find the best parameters
# best_params = pipeline.perform_grid_search(X_train, y_train)

# # Evaluate the pipeline on the test set
# test_accuracy = pipeline.evaluate(X_test, y_test)
