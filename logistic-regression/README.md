# Logistic Regression from Scratch – Breast Cancer Classification

## 🧠 Problem Statement

This project solves a **binary classification problem** using **logistic regression implemented from scratch**. The goal is to classify whether a tumor is **malignant (cancerous)** or **benign (non-cancerous)** based on features extracted from cell nuclei images.

The dataset used is the **Breast Cancer Wisconsin dataset**, which is commonly applied in machine learning and medical diagnosis problems.

---

## 🧩 Approach

The algorithm implements **binary logistic regression** trained via **gradient descent**. It does **not** rely on high-level libraries like scikit-learn for model training — the learning algorithm is fully implemented in NumPy.

The model:
- Adds a bias term (intercept) manually.
- Uses the **sigmoid activation function** to convert linear outputs into probabilities.
- Minimizes the **logistic loss function** using a basic **gradient descent optimizer**.
- Predicts binary labels by applying a probability threshold (default: 0.5).

---

## 📐 Mathematical Foundations

- **Sigmoid Function**:  
  \[
  \sigma(z) = \frac{1}{1 + e^{-z}}
  \]
  Converts linear outputs into probabilities between 0 and 1.

- **Logistic Loss Gradient**:  
  \[
  \nabla J(\theta) = \frac{1}{m} X^T (\sigma(X\theta) - y)
  \]
  Used to update model parameters via gradient descent.

---

## ⚙️ Training and Evaluation

- The input data is standardized using **z-score normalization**.
- The dataset is split into a **training set (80%)** and a **test set (20%)**.
- Model performance is evaluated using **accuracy** on both sets.

### Example results:
```

Training Accuracy: 95.16%
Test Accuracy:    97.37%

````

These results show that the model generalizes well, with **no signs of overfitting**.

---

## 📦 Dependencies

- Python 3.x
- NumPy
- scikit-learn (only for dataset loading and metrics)

---

## 🚀 How to Run

```bash
python main.py
````

Make sure the `main.py` file contains the code for training, prediction, and evaluation.

---

## ✅ Summary

This project demonstrates how to implement **logistic regression from scratch** for medical diagnosis. Despite its simplicity, the algorithm achieves high accuracy, showing the effectiveness of logistic regression as a baseline method for binary classification tasks.

```