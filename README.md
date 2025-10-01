# Machine Learning Fundamentals

This repository provides clear explanations of fundamental concepts in Machine Learning.  
The goal is to create a structured and beginner-friendly resource that covers the most important building blocks of ML models and techniques.

---

## Table of Contents

- Overfitting vs Underfitting  
- Bias-Variance Tradeoff  
- Gradient Descent  
  - What is Gradient Descent?  
  - Types of Gradient Descent (Batch, Stochastic, Mini-Batch)  
  - Learning Rate and its effect  
- Loss Functions  
  - MSE, MAE, Hinge Loss  
- Distance Measures  
  - Euclidean Distance  
  - Minkowski Distance  
  - Cosine Distance/Similarity  
- Regularization  
- Softmax and Cost Function  

---

## Concepts

### Overfitting vs Underfitting

- **Underfitting:**  
  Occurs when a model is too simple to capture the underlying pattern in the data. It performs poorly on both training and test data.  
  **Example:** Using a linear model to fit highly non-linear data.

- **Overfitting:**  
  Occurs when a model learns the noise in the training data instead of the true pattern. It performs very well on training data but poorly on unseen test data.  
  **Example:** A very deep neural network trained on a small dataset.

- **Solution strategies:**  
  - Regularization (L1, L2)  
  - Early stopping  
  - Using more data  
  - Simplifying the model  

---

### Bias-Variance Tradeoff

- **Bias:** Error due to overly simple models that cannot capture the true relationship.  
- **Variance:** Error due to models being too sensitive to training data (overfitting).  

**Goal:** Find a balance between bias and variance to minimize total error (generalization error).  
- High bias → underfitting  
- High variance → overfitting  

---

### Gradient Descent

#### What is Gradient Descent?

Gradient Descent is an optimization algorithm used to minimize a **loss function** by iteratively updating the model parameters in the direction of the steepest decrease of the loss.

#### Types of Gradient Descent

1. **Batch Gradient Descent:** Uses the entire training set to compute the gradient.  
   - Pros: Stable convergence  
   - Cons: Slow on large datasets

2. **Stochastic Gradient Descent (SGD):** Uses one training example at a time.  
   - Pros: Faster, can escape local minima  
   - Cons: Noisy updates

3. **Mini-Batch Gradient Descent:** Uses small batches of training examples.  
   - Pros: Combines benefits of batch and SGD  
   - Most commonly used in practice  

#### Learning Rate

- The learning rate ($\alpha$) controls the step size of each update.  
- Too small → slow convergence  
- Too large → may overshoot or diverge  

---

### Loss Functions

- **MSE (Mean Squared Error):** Measures average squared difference between predicted and actual values.  
- **MAE (Mean Absolute Error):** Measures average absolute difference between predicted and actual values.  
- **Hinge Loss:** Commonly used in SVMs, penalizes misclassified points and points within the margin.  

---

### Distance Measures

- **Euclidean Distance:** Standard straight-line distance between two points.  
- **Minkowski Distance:** Generalization of Euclidean and Manhattan distances.  
- **Cosine Distance/Similarity:** Measures the angle between two vectors, useful for text or high-dimensional data.  

---

### Regularization

Regularization is used to **prevent overfitting** by adding a penalty term to the loss function.

- **L1 Regularization (Lasso):** Adds absolute value of weights. Can lead to sparse weights (feature selection).  
- **L2 Regularization (Ridge):** Adds squared value of weights. Reduces magnitude of weights but rarely zeros them.  
- **Elastic Net:** Combination of L1 and L2.  

---

### Softmax and Cost Function

- **Softmax:** Converts raw scores (logits) into probabilities that sum to 1. Used in multi-class classification.  

$$
\hat{y}_j = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}
$$

- **Cross-Entropy Loss:** Measures difference between predicted probability distribution ($\hat{y}$) and true labels ($y$).  

$$
J(\hat{y}, y) = - \sum_{j=1}^{K} y_j \log(\hat{y}_j)
$$

- **Gradient for Softmax + Cross-Entropy:**  

$$
\frac{\partial J}{\partial z_j} = \hat{y}_j - y_j
$$

This simple gradient allows efficient backpropagation for training neural networks.

---

##  License
This project is licensed under the [MIT License](LICENSE).  
