# Support-Vector Networks Algorithm Implementation

This repository contains a Python implementation of the SVM algorithm. The models are built from scratch without relying on any machine learning libraries, showcasing the underlying principles and algorithms.

## Description

In this project, I explore a basic application of the Support Vector Machine (SVM) algorithm for a two-group classification problem using a linear kernel. The SVM algorithm is implemented from scratch to separate features in a two-dimensional space. The model aims to find the optimal hyperplane that maximizes the margin between two classes. Through this project, I demonstrate the principles of SVM, including the use of regularization and gradient descent for training the model.

## Mathematical Formulas for SVM

1. **Condition for Update**:
   $$y_i \cdot (\mathbf{x}_i \cdot \mathbf{w} - b) \geq 1$$

2. **Weights Update (if condition is met)**:
   $$\mathbf{w} = \mathbf{w} - \eta \cdot (2 \cdot \lambda \cdot \mathbf{w})$$

3. **Weights and Bias Update (if condition is not met)**:
   $$\mathbf{w} = \mathbf{w} - \eta \cdot (2 \cdot \lambda \cdot \mathbf{w} - \mathbf{x}_i \cdot y_i)$$
   
   $$b = b - \eta \cdot y_i$$

4. **Linear Output for Prediction**:
   $$\text{linear\_output} = X \cdot \mathbf{w} + b$$

5. **Prediction**:
   $$\hat{y} = \text{sign}(\text{linear\_output})$$


## Files

- `svm_2D_linear_seperable_test.ipynb`: A Jupyter Notebook demonstrating the SVM algorithm.
- `svn.py`: A Python script containing the implementation of the SVM algorithm from scratch.
- `requirements.txt`: List of required libraries and dependencies.
- `README.md`: Documentation of the project.

## References
