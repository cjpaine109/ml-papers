# Support-Vector Networks Algorithm Implementation

This repository contains a Python implementation of the SVM algorithm. The models are built from scratch without relying on any machine learning libraries, showcasing the underlying principles and algorithms.

## Description

In this project, I explore a basic application of the Support Vector Machine (SVM) algorithm for a two-group classification problem using a linear kernel. The SVM algorithm is implemented from scratch to separate features in a two-dimensional space. The model aims to find the optimal hyperplane that maximizes the margin between two classes. Through this project, I demonstrate the principles of SVM, including the use of regularization and gradient descent for training the model.

## Mathematical Formulas for SVM

1. **Margins**:
   $M = \frac{2}{\|\mathbf{w}\|}$

2. **Main Hyperplane**:
   $\mathbf{w} \cdot \mathbf{x} - b = 0$

3. **Cost Function (Hinge Loss)**:
   $L(\mathbf{w}, b) = \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \max(0, 1 - y_i (\mathbf{x}_i \cdot \mathbf{w} - b))$


## Files

- `svm_2D_linear_seperable_test.ipynb`: A Jupyter Notebook demonstrating the SVM algorithm.
- `svn.py`: A Python script containing the implementation of the SVM algorithm from scratch.
- `requirements.txt`: List of required libraries and dependencies.
- `README.md`: Documentation of the project.

## References

other -> https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
