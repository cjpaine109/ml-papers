# =============================================================================
#                           Support-Vector Networks Algorithm Implementation
# =============================================================================
# Reference: "Support-Vector Networks" by Corinna Cortes & Vladimir Vapnik
# Author:    Conner Paine
# Date:      07/02/2024
# =============================================================================

# IMPORTS
import numpy as np
from typing import List

class SVM:
    """
    Support Vector Machine (SVM) classifier.

    Attributes:
        learning_rate (float): Learning rate for gradient descent.
        lambda_param (float): Regularization parameter.
        epochs (int): Number of iterations over the training dataset.
        weights (ndarray): Weight vector.
        bias (float): Bias term.
    """

    def __init__(self, learning_rate: float = 0.001, lambda_param: float = 0.01, epochs: int = 1000):
        """
        Initializes the SVM with specified parameters.

        Args:
            learning_rate (float): Learning rate for gradient descent.
            lambda_param (float): Regularization parameter.
            epochs (int): Number of iterations over the training dataset.
        """
        self.lr_ = learning_rate
        self.lambda_param_ = lambda_param
        self.epochs_ = epochs
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the SVM classifier on the given dataset.

        Args:
            X (ndarray): Training feature dataset.
            y (ndarray): Training labels.
        """
        n_samples, n_features = X.shape

        # Convert dichotomous output variable y (0, 1) -> (-1, 1)
        y_ = np.where(y <= 0, -1, 1)

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs_):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr_ * (2 * self.lambda_param_ * self.weights)
                else:
                    self.weights -= self.lr_ * (
                        2 * self.lambda_param_ * self.weights - np.dot(x_i, y_[idx])
                    )
                    self.bias -= self.lr_ * y_[idx]

    def predict(self, X: np.ndarray) -> List[float]:
        """
        Predicts the labels for the given dataset.

        Args:
            X (ndarray): Feature dataset to predict.

        Returns:
            List[float]: Predicted labels.
        """
        linear_output = X @ self.weights + self.bias
        return np.sign(linear_output).tolist()