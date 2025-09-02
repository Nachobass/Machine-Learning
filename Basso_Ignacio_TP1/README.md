# TP1 - Linear Regression and Feature Engineering

## Overview
This practical work focuses on **linear regression models** and **feature engineering** for house price prediction. The project implements multiple regression algorithms from scratch and explores different regularization techniques.

## Dataset
- **Domain**: Real estate price prediction
- **Files**: 
  - `casas_dev.csv` - Development dataset for training and validation
  - `casas_test.csv` - Test dataset for final predictions
  - `vivienda_Amanda.csv` - Additional reference data

## Implemented Models
- **Linear Regression** (Pseudo-inverse method)
- **Regularized Linear Regression** (Ridge/L2 regularization)
- **Gradient Descent** implementation
- **Lasso Regression** (L1 regularization with gradient descent)

## Key Features
- Custom implementations of linear regression algorithms
- Feature engineering and preprocessing pipelines
- Missing value imputation using statistical methods
- Data normalization and standardization
- Model performance visualization and evaluation

## Project Structure
```
├── data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Preprocessed datasets
├── notebooks/
│   └── Entrega_TP1.ipynb      # Main analysis notebook
└── src/
    ├── data_splitting.py       # Train/validation splitting
    ├── metrics.py              # Evaluation metrics
    ├── models.py               # Linear regression implementations
    ├── preprocessing.py        # Data preprocessing functions
    └── utils.py                # Utility functions
```

## Dependencies
Create a `requirements.txt` file with:
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
jupyter>=1.0.0
```

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the main notebook: `jupyter notebook notebooks/Entrega_TP1.ipynb`
3. Or import modules directly:
```python
from src.models import LinearRegression
from src.preprocessing import handle_missing_values, normalize_data
```

## Key Learning Objectives
- Understand linear regression mathematics and implementation
- Learn feature engineering techniques for real estate data
- Practice regularization methods (L1/L2)
- Implement gradient descent optimization
- Handle missing data and outliers

## Results
The project demonstrates the effectiveness of different regression approaches and the importance of proper feature engineering in house price prediction tasks.
