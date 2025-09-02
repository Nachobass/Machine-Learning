# TP3 - Deep Learning and Neural Networks

## Overview
This practical work focuses on **deep learning** and **neural network** architectures using PyTorch. The project implements custom neural networks for image classification tasks and explores different architectures and optimization techniques.

## Dataset
- **Domain**: Computer Vision - Image Classification
- **Format**: NumPy arrays containing image data
- **Files**:
  - `X_images.npy` - Image features/pixel data
  - `y_images.npy` - Image labels
  - `X_COMP.npy` - Competition/test data

## Implemented Models
- **M1 Model**: Custom feedforward neural network with configurable layers
- **Deep Neural Networks** with multiple hidden layers
- **ReLU activation** functions
- **He initialization** for weights
- **Custom PyTorch implementations**

## Key Features
- Custom neural network implementation using PyTorch
- Flexible architecture with configurable layer dimensions
- He initialization for improved convergence with ReLU networks
- Training loops with progress tracking (tqdm)
- Model evaluation and prediction capabilities
- Data preprocessing and splitting utilities

## Project Structure
```
├── data/
│   ├── raw/                   # Original numpy datasets
│   └── processed/             # Preprocessed datasets
├── notebooks/
│   └── Basso_Ignacio_Notebook_TP3.ipynb  # Main analysis notebook
├── src/
│   ├── data_splitting.py      # Train/validation splitting
│   ├── pytorch_models2.py     # Neural network models
│   └── utils2.py              # Utility functions
├── Basso_Ignacio_Informe_TP3.pdf
├── Basso_Ignacio_predicciones.csv  # Final predictions
└── template_informe.tex       # LaTeX template
```

## Dependencies
Create a `requirements.txt` file with:
```
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
tqdm>=4.62.0
scikit-learn>=1.0.0
jupyter>=1.0.0
```

## Model Architecture
The M1 model features:
- **Configurable depth**: Specify layer dimensions as a list
- **ReLU activations**: Between hidden layers
- **He initialization**: Optimized for ReLU networks
- **Linear output layer**: For classification/regression tasks

### Example Usage:
```python
from src.pytorch_models2 import M1

# Define a 3-layer network: input(784) -> hidden(128) -> hidden(64) -> output(10)
layer_dims = [784, 128, 64, 10]
model = M1(layer_dims)

# Forward pass
output = model(input_tensor)
```

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Ensure PyTorch is properly installed for your system
3. Run the main notebook: `jupyter notebook notebooks/Basso_Ignacio_Notebook_TP3.ipynb`
4. Or import models directly:
```python
from src.pytorch_models2 import M1
from src.utils2 import *
```

## Key Learning Objectives
- Understand deep neural network architectures
- Learn PyTorch framework for deep learning
- Implement custom neural networks from scratch
- Practice weight initialization techniques
- Master training loops and optimization
- Handle image data preprocessing and augmentation

## Technical Details
- **Framework**: PyTorch
- **Activation**: ReLU for hidden layers
- **Initialization**: He/Kaiming normal initialization
- **Optimization**: Gradient-based optimization (Adam/SGD)
- **Architecture**: Feedforward neural networks

## Results
The project demonstrates the implementation and training of deep neural networks for image classification, showcasing the power of custom PyTorch implementations and proper initialization strategies for achieving good convergence and performance.
