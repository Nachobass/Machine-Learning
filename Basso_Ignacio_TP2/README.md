# TP2 - Classification Problems

## Overview
This practical work focuses on **binary and multi-class classification** using logistic regression. The project includes two distinct classification problems with different characteristics and challenges.

## Problems

### Problem 1: Cell Diagnosis Classification
- **Domain**: Medical diagnosis - breast cancer cell classification
- **Type**: Binary classification (Malignant vs Benign)
- **Dataset**: Cell diagnosis with multiple cellular features
- **Challenge**: Class imbalance handling

### Problem 2: Baseball WAR Classification  
- **Domain**: Sports analytics - baseball player performance
- **Type**: Multi-class classification
- **Dataset**: Baseball player statistics for WAR (Wins Above Replacement) prediction
- **Challenge**: Multi-class prediction with performance metrics

## Implemented Models
- **Logistic Regression with L2 Regularization** (from scratch)
- **One-vs-Rest (OvR)** approach for multi-class problems
- **Softmax/Multinomial** logistic regression
- **Class weighting** for imbalanced datasets

## Key Features
- Custom logistic regression implementation with gradient descent
- Sigmoid and softmax activation functions
- L2 regularization to prevent overfitting
- Class imbalance handling through weighted loss functions
- Multiple strategies for multi-class classification
- Comprehensive evaluation metrics

## Project Structure
```
├── Problema 1/                # Cell diagnosis classification
│   ├── data/
│   │   ├── raw/               # Original datasets
│   │   └── processed/         # Preprocessed datasets
│   ├── notebooks/
│   │   └── Basso_Ignacio_Notebook_TP2_1.ipynb
│   └── src/                   # Source code for Problem 1
├── Problema 2/                # Baseball WAR classification
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   ├── notebooks/
│   │   └── Basso_Ignacio_Notebook_TP2_2.ipynb
│   └── src/                   # Source code for Problem 2
├── Basso_Ignacio_Informe_TP2.pdf
└── template_informe.tex       # LaTeX template for report
```

## Dependencies
Create a `requirements.txt` file with:
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
```

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. For Problem 1: `jupyter notebook "Problema 1/notebooks/Basso_Ignacio_Notebook_TP2_1.ipynb"`
3. For Problem 2: `jupyter notebook "Problema 2/notebooks/Basso_Ignacio_Notebook_TP2_2.ipynb"`

## Key Learning Objectives
- Understand logistic regression mathematics and implementation
- Learn binary and multi-class classification techniques
- Handle class imbalance in real-world datasets
- Implement gradient descent for logistic regression
- Compare different multi-class strategies (OvR vs Softmax)
- Evaluate classification performance with appropriate metrics

## Key Algorithms
- **Sigmoid Function**: For binary classification probabilities
- **Softmax Function**: For multi-class probability distributions
- **Gradient Descent**: Optimization with L2 regularization
- **One-vs-Rest**: Binary classification extended to multi-class
- **Class Weighting**: Handling imbalanced datasets

## Results
The project demonstrates effective classification approaches for both medical diagnosis and sports analytics, highlighting the importance of proper evaluation metrics and class imbalance handling in real-world classification problems.
