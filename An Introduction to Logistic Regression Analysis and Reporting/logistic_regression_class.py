# =============================================================================
#                           Logistic Regression Model
# =============================================================================
# Reference: "An Introduction to Logistic Regression Analysis and Reporting"
#            by Chao-Ying Joanne Peng, Kuk Lida Lee, and Gary M. Ingersoll
# Author:    Conner Paine
# Date:      06/20/2024
# =============================================================================

# IMPORTS
import numpy as np

class LogisticRegression:
    """
    A simple implementation of the Logistic Regression algorithm.
    """

    def __init__(self):
        """
        Initializes the Logistic Regression model.
        """
        self.theta = None

    @staticmethod
    def add_intercept(X):
        """
        Adds an intercept term to the feature matrix X.

        Parameters:
        X (np.ndarray): Feature matrix.

        Returns:
        np.ndarray: Feature matrix with an intercept term.
        """
        return np.hstack([np.ones((X.shape[0], 1)), X])
    
    @staticmethod
    def logistic_function(z):
        """
        Computes the logistic function.

        Parameters:
        z (np.ndarray): Input value/s.

        Returns:
        np.ndarray: Output of logistic function applied to input.
        """
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y, *, n_epochs: int = 1000, lr: float = 0.01) -> None:
        """
        Fits the Logistic Regression model to the training data.

        Parameters:
        X (np.ndarray): Training feature matrix.
        y (np.ndarray): Training labels.
        n_epochs (int, optional): Number of epochs for gradient descent. Default is 1000.
        lr (float, optional): Learning rate for gradient descent. Default is 0.01.
        """
        X = self.add_intercept(X)
        y = y.reshape(-1, 1)  # Ensure y is a column vector
        m, n = X.shape
        self.theta = np.random.randn(n, 1)

        for epoch in range(n_epochs):
            z = X @ self.theta
            h = self.logistic_function(z)
            error = h - y
            gradient = (X.T @ error) / m
            self.theta -= lr * gradient

        print(f'Final coefficients: {self.theta.ravel()}')

    def predict_prob(self, X):
        """
        Predicts the probability of the positive class for input features.

        Parameters:
        X (np.ndarray): Feature matrix.

        Returns:
        np.ndarray: Probability predictions.
        """
        X = self.add_intercept(X)
        return self.logistic_function(X @ self.theta)

    def predict(self, X, threshold=0.5):
        """
        Predicts the class labels for input features.

        Parameters:
        X (np.ndarray): Feature matrix.
        threshold (float, optional): Decision threshold. Default is 0.5.

        Returns:
        np.ndarray: Class label predictions.
        """
        probs = self.predict_prob(X)
        return (probs >= threshold).astype(int)
