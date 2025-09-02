import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import rgb_to_hsv


def load_data_reduce_dimensionality():
    """
    Loads the MNIST dataset from a CSV file, separates features and labels, and optionally plots the class distribution.

    Reads the dataset from '../data/raw/MNIST_dataset.csv'. If a 'label' column is present, it is separated from the feature data.
    If labels are available, displays a bar plot showing the distribution of digit classes in the dataset.

    Returns:
        X (np.ndarray): Feature matrix containing the pixel values of the images.
        labels (np.ndarray or None): Array of digit labels if present in the dataset, otherwise None.
    """
    df = pd.read_csv("../data/raw/MNIST_dataset.csv")
    if 'label' in df.columns:
        labels = df['label'].values
        X = df.drop(columns=['label']).values
    else:
        X = df.values
        labels = None

    if labels is not None:
        unique_labels, counts = np.unique(labels, return_counts=True)
        plt.figure(figsize=(10, 5))
        plt.bar(unique_labels, counts)
        plt.title("Distribución de clases en el dataset MNIST")
        plt.xlabel("Dígitos")
        plt.ylabel("Número de muestras")
        plt.xticks(unique_labels)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
    return X, labels

def pca_with_svd(X, X_centered):
    """
    Performs Principal Component Analysis (PCA) on the given data matrix using Singular Value Decomposition (SVD).

    Parameters
    ----------
    X : np.ndarray
        The original data matrix of shape (n_samples, n_features).
    X_centered : np.ndarray
        The mean-centered data matrix of shape (n_samples, n_features).

    Returns
    -------
    U : np.ndarray
        Left singular vectors of the centered data matrix.
    S : np.ndarray
        Singular values of the centered data matrix.
    Vt : np.ndarray
        Right singular vectors (transposed) of the centered data matrix.
    explained_variance : np.ndarray
        The amount of variance explained by each principal component.
    explained_variance_ratio : np.ndarray
        The proportion of variance explained by each principal component.
    cumulative_variance_ratio : np.ndarray
        The cumulative proportion of variance explained by the principal components.
    """
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    explained_variance = (S**2) / (X.shape[0] - 1)
    total_variance = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_variance
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    return U, S, Vt, explained_variance, explained_variance_ratio, cumulative_variance_ratio

def plot_error_vs_components(components_range, errors):
    """
    Plots the reconstruction mean squared error (ECM) as a function of the number of principal components.

    Args:
        components_range (array-like): Sequence of the number of principal components used.
        errors (array-like): Corresponding reconstruction mean squared errors for each number of components.

    Displays:
        A line plot showing how the reconstruction error changes with the number of principal components.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(components_range, errors, marker='o')
    plt.title("ECM de reconstrucción vs número de componentes")
    plt.xlabel("Número de componentes principales")
    plt.ylabel("Error cuadrático medio (ECM)")
    plt.grid(True)
    plt.show()

def plot_cumulative_variance(cumulative_variance_ratio, thresholds):
    """
    Plots the cumulative explained variance ratio for principal components and highlights specified variance thresholds.

    Parameters
    ----------
    cumulative_variance_ratio : array-like
        A 1D array or list containing the cumulative explained variance ratios for each principal component.
    thresholds : list or array-like
        A list of float values between 0 and 1 representing the cumulative variance thresholds to highlight on the plot.

    Returns
    -------
    components_for_thresholds : dict
        A dictionary mapping each threshold to the minimum number of principal components required to reach or exceed that threshold.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_variance_ratio[:350], marker='.', label='Varianza acumulada', color='blue')

    cmap = cm.Reds
    norm = mcolors.Normalize(vmin=0.6, vmax=1.0)

    components_for_thresholds = {}
    for t in thresholds:
        k = np.argmax(cumulative_variance_ratio >= t) + 1
        color = cmap(norm(t))
        components_for_thresholds[t] = k

        plt.axhline(y=t, color=color, linestyle='--', linewidth=1.5)
        plt.axvline(x=k, color=color, linestyle=':', linewidth=1.5)

        hsv = rgb_to_hsv(np.array(color[:3]).reshape(1, 1, 3))[0, 0]
        brightness = hsv[2]
        text_color = 'black' if brightness > 0.6 else 'white'

        plt.annotate(f'{int(t*100)}%\n{k} comps',
                    xy=(k, t),
                    xytext=(k + 10, t - 0.05),
                    arrowprops=dict(arrowstyle="->", color=color),
                    bbox=dict(boxstyle="round,pad=0.3", fc=color, alpha=0.7),
                    fontsize=9,
                    color=text_color)

    plt.title("Varianza explicada acumulada")
    plt.xlabel("Número de componentes principales")
    plt.ylabel("Varianza explicada acumulada")
    plt.grid(True)
    plt.xlim(0, 350)
    plt.ylim(0, 1.01)
    plt.tight_layout()
    plt.show()
    return components_for_thresholds