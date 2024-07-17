# Support-Vector Networks Algorithm Implementation

This repository contains a Python implementation of the SVM algorithm. The models are built from scratch without relying on any machine learning libraries, showcasing the underlying principles and algorithms.

## Description

In this project, I explore a basic application of the Support Vector Machine (SVM) algorithm for a two-group classification problem using a linear kernel. The SVM algorithm is implemented from scratch to separate features in a two-dimensional space. The model aims to find the optimal hyperplane that maximizes the margin between two classes. Through this project, I demonstrate the principles of SVM, including the use of regularization and gradient descent for training the model.

## Mathematical Formulas for SVM

### Margins
The margin \( M \) is defined as:
$M = \frac{2}{\|\mathbf{w}\|}$

### Inequality Conditions for Margins
For a feature vector \(\mathbf{x}_i\) and its label \(y_i\):
$y_i (\mathbf{w} \cdot \mathbf{x}_i - b) \geq 1$

### Main Hyperplane
The equation for the main hyperplane is:
$\mathbf{w} \cdot \mathbf{x} - b = 0$

### Cost Function (Hinge Loss)
The hinge loss cost function \( L(\mathbf{w}, b) \) is given by:
$L(\mathbf{w}, b) = \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \max(0, 1 - y_i (\mathbf{x}_i \cdot \mathbf{w} - b))$

### Explanation of the Formulas:
- $M$ represents the margin, which is inversely proportional to the norm of the weight vector \(\mathbf{w}\).
- $y_i$ is the label of the \(i\)-th training example.
- \(\mathbf{x}_i\) is the feature vector of the \(i\)-th training example.
- \(\mathbf{w}\) represents the weights or parameters of the model.
- $b$ is the bias term.
- $C$ is the regularization parameter.
- $\max(0, 1 - y_i (\mathbf{x}_i \cdot \mathbf{w} - b))$ represents the hinge loss for the \(i\)-th training example.

## Files

- `svm_2D_linear_seperable_test.ipynb`: A Jupyter Notebook demonstrating the SVM algorithm.
- `svn.py`: A Python script containing the implementation of the SVM algorithm from scratch.
- `requirements.txt`: List of required libraries and dependencies.
- `README.md`: Documentation of the project.

## References

other -> https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
