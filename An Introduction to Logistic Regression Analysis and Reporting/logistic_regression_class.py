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
    def __init__(self):
        self.theta = None

    @staticmethod
    def add_intercept(X):
        return np.hstack([np.ones((X.shape[0], 1)), X])
    
    @staticmethod
    def logistic_function(z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y, *, n_epochs: int = 1000, lr: float = 0.01) -> None:
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
        X = self.add_intercept(X)
        return self.logistic_function(X @ self.theta)

    def predict(self, X, threshold=0.5):
        probs = self.predict_prob(X)
        return (probs >= threshold).astype(int)