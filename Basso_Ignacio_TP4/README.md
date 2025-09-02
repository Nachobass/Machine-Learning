# TP4 - Unsupervised Learning and Clustering

## Overview
This practical work focuses on **unsupervised learning** algorithms, specifically **clustering techniques** and **dimensionality reduction**. The project implements multiple clustering algorithms from scratch and explores their applications on different datasets.

## Datasets
- **Clustering Dataset**: General clustering data (`clustering.csv`)
- **MNIST Dataset**: Handwritten digits for clustering analysis (`MNIST_dataset.csv.zip` - needs decompression)

## Implemented Algorithms

### Clustering Methods
- **K-Means**: Custom implementation with multiple initialization strategies
- **DBSCAN**: Density-based clustering for arbitrary shaped clusters
- **Gaussian Mixture Models (GMM)**: Probabilistic clustering with EM algorithm

### Dimensionality Reduction
- **Principal Component Analysis (PCA)**: For data visualization and preprocessing
- **Variational Autoencoder (VAE)**: Deep learning approach to dimensionality reduction

## Key Features
- Complete implementations of clustering algorithms from scratch
- Multiple evaluation metrics (silhouette score, elbow method, etc.)
- Visualization tools for cluster analysis
- Hyperparameter optimization and model selection
- Comparative analysis of different clustering approaches

## Project Structure
```
├── data/
│   └── raw/                   # Original datasets
├── notebooks/
│   └── Basso_Ignacio_Notebook_TP4.ipynb  # Main analysis notebook
├── results/                   # Basic clustering results
├── results_kmeans/           # K-means specific results
├── results_kmeans_silhouette/ # K-means with silhouette optimization
├── results_gmm_vectorized/   # GMM results
├── src/
│   ├── data_splitting.py     # Data preprocessing
│   ├── kmeans.py            # K-means implementation
│   ├── dbscan.py            # DBSCAN implementation
│   ├── gmm_implementation.py # Gaussian Mixture Models
│   ├── pca.py               # Principal Component Analysis
│   ├── vae.py               # Variational Autoencoder
│   └── utils.py             # Utility functions
├── Basso_Ignacio_Informe_TP4.pdf
└── template_informe.tex      # LaTeX template
```

## Dependencies
Create a `requirements.txt` file with:
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
torch>=1.12.0
torchvision>=0.13.0
tqdm>=4.62.0
jupyter>=1.0.0
scipy>=1.7.0
```

## Algorithm Details

### K-Means Clustering
- Multiple initialization strategies
- Elbow method for optimal k selection
- Silhouette analysis for cluster quality
- Custom distance metrics

### DBSCAN
- Density-based clustering
- Noise point detection
- Parameter optimization (eps, min_samples)
- Arbitrary cluster shape handling

### Gaussian Mixture Models
- Expectation-Maximization algorithm
- Probabilistic cluster assignments
- Model selection with information criteria
- Covariance matrix estimation

### PCA
- Eigenvalue decomposition
- Variance explained analysis
- Data visualization in reduced dimensions
- Feature selection capabilities

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. **Decompress the MNIST dataset**: Extract `MNIST_dataset.csv` from `data/raw/MNIST_dataset.csv.zip`
3. Run the main notebook: `jupyter notebook notebooks/Basso_Ignacio_Notebook_TP4.ipynb`
4. Or import specific algorithms:
```python
from src.kmeans import KMeans
from src.dbscan import DBSCAN
from src.gmm_implementation import GaussianMixture
from src.pca import PCA
```

## Key Learning Objectives
- Understand unsupervised learning principles
- Implement clustering algorithms from scratch
- Learn dimensionality reduction techniques
- Practice model evaluation for unsupervised methods
- Compare different clustering approaches
- Visualize high-dimensional data

## Evaluation Metrics
- **Silhouette Score**: Cluster cohesion and separation
- **Elbow Method**: Optimal number of clusters
- **Inertia/WCSS**: Within-cluster sum of squares
- **AIC/BIC**: Information criteria for model selection
- **Visual Inspection**: Cluster visualization and interpretation

## Results Organization
Results are organized in separate directories:
- `results/`: Basic clustering outputs
- `results_kmeans/`: K-means centroids and labels
- `results_kmeans_silhouette/`: Silhouette-optimized results
- `results_gmm_vectorized/`: GMM parameters and statistics

## Results
The project provides a comprehensive comparison of different unsupervised learning approaches, demonstrating their strengths and weaknesses on various types of data. Special attention is given to parameter selection and evaluation strategies for unsupervised methods.
