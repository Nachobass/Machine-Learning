import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from typing import List, Optional
from scipy.stats import multivariate_normal
from kmeans import kmeans, preprocess_data, load_data
import json
import os


def initialize_gmm_parameters(X: np.ndarray, k: int):
    """
    Initializes the parameters for a Gaussian Mixture Model (GMM) using K-means clustering.

    Parameters:
        X (np.ndarray): The input data array of shape (n_samples, n_features).
        k (int): The number of Gaussian components (clusters).

    Returns:
        means (np.ndarray): Array of shape (k, n_features) containing the initial means for each component.
        covariances (np.ndarray): Array of shape (k, n_features, n_features) containing the initial covariance matrices for each component.
        weights (np.ndarray): Array of shape (k,) containing the initial weights (mixing coefficients) for each component.
    """
    labels, means, _ = kmeans(X, k,X_mean=np.mean(X, axis=0), X_std=np.std(X, axis=0))
    n_samples, n_features = X.shape
    weights = np.zeros(k)
    covariances = np.zeros((k, n_features, n_features))

    for i in range(k):
        cluster_points = X[labels == i]
        weights[i] = len(cluster_points) / n_samples
        if len(cluster_points) > 1:
            covariances[i] = np.cov(cluster_points, rowvar=False)
        else:
            covariances[i] = np.eye(n_features)

    return means, covariances, weights

def compute_responsibilities(X: np.ndarray, means: np.ndarray, covariances: np.ndarray, weights: np.ndarray):
    """
    Compute the responsibilities (posterior probabilities) for each data point and each Gaussian component in a Gaussian Mixture Model (GMM).

    Parameters
    ----------
    X : np.ndarray
        Array of shape (n_samples, n_features) containing the data points.
    means : np.ndarray
        Array of shape (n_components, n_features) containing the mean vectors of each Gaussian component.
    covariances : np.ndarray
        Array of shape (n_components, n_features, n_features) containing the covariance matrices of each Gaussian component.
    weights : np.ndarray
        Array of shape (n_components,) containing the mixture weights for each component.

    Returns
    -------
    responsibilities : np.ndarray
        Array of shape (n_samples, n_components) containing the responsibility of each component for each data point.
    """
    n_samples = X.shape[0]
    k = means.shape[0]
    responsibilities = np.zeros((n_samples, k))

    for i in range(k):
        rv = multivariate_normal(mean=means[i], cov=covariances[i])
        responsibilities[:, i] = weights[i] * rv.pdf(X)

    responsibilities_sum = responsibilities.sum(axis=1)[:, np.newaxis]
    responsibilities /= responsibilities_sum

    return responsibilities

def update_parameters(X: np.ndarray, responsibilities: np.ndarray):
    """
    Updates the parameters (means, covariances, and weights) of a Gaussian Mixture Model (GMM) given the data and current responsibilities.

    Args:
        X (np.ndarray): The input data of shape (n_samples, n_features).
        responsibilities (np.ndarray): The responsibility matrix of shape (n_samples, n_components), where each entry represents the probability of a sample belonging to a component.

    Returns:
        means (np.ndarray): Updated means of shape (n_components, n_features).
        covariances (np.ndarray): Updated covariance matrices of shape (n_components, n_features, n_features).
        weights (np.ndarray): Updated mixture weights of shape (n_components,).
    """
    n_samples, n_features = X.shape
    k = responsibilities.shape[1]

    Nk = responsibilities.sum(axis=0)
    weights = Nk / n_samples
    means = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]
    covariances = np.zeros((k, n_features, n_features))

    for i in range(k):
        diff = X - means[i]
        weighted_diff = responsibilities[:, i][:, np.newaxis] * diff
        covariances[i] = np.dot(weighted_diff.T, diff) / Nk[i]
        # numerical stability: add small value to diagonal
        covariances[i] += 1e-6 * np.eye(n_features)

    return means, covariances, weights

def log_likelihood(X, means, covariances, weights):
    """
    Computes the log-likelihood of the data under a Gaussian Mixture Model (GMM).

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        The input data.
    means : np.ndarray of shape (n_components, n_features)
        The mean vectors of each Gaussian component.
    covariances : np.ndarray of shape (n_components, n_features, n_features)
        The covariance matrices of each Gaussian component.
    weights : np.ndarray of shape (n_components,)
        The mixture weights for each component.

    Returns
    -------
    float
        The total log-likelihood of the data under the GMM.
    """
    k = means.shape[0]
    n_samples = X.shape[0]
    log_probs = np.zeros((n_samples, k))

    for i in range(k):
        log_probs[:, i] = np.log(weights[i] + 1e-10) + multivariate_normal.logpdf(X, mean=means[i], cov=covariances[i])

    # use log-sum-exp trick for numerical stability
    max_log_probs = np.max(log_probs, axis=1, keepdims=True)
    stabilized = np.exp(log_probs - max_log_probs)
    log_likelihoods = max_log_probs + np.log(np.sum(stabilized, axis=1, keepdims=True))

    return np.sum(log_likelihoods)

def gmm(X: np.ndarray, k: int, max_iters: int = 100, tol: float = 1e-4):
    """
    Fits a Gaussian Mixture Model (GMM) to the data using the Expectation-Maximization (EM) algorithm.

    Parameters:
        X (np.ndarray): Input data of shape (n_samples, n_features).
        k (int): Number of Gaussian components (clusters).
        max_iters (int, optional): Maximum number of EM iterations. Default is 100.
        tol (float, optional): Convergence threshold for the change in log-likelihood. Default is 1e-4.

    Returns:
        labels (np.ndarray): Cluster assignments for each data point (shape: (n_samples,)).
        means (np.ndarray): Means of the Gaussian components (shape: (k, n_features)).
        covariances (np.ndarray): Covariance matrices of the Gaussian components (shape: (k, n_features, n_features)).
        current_ll (float): Final log-likelihood value after convergence or reaching max_iters.
    """
    means, covariances, weights = initialize_gmm_parameters(X, k)
    prev_log_likelihood = -np.inf

    for iteration in range(max_iters):
        # E-step
        responsibilities = compute_responsibilities(X, means, covariances, weights)
        # M-step
        means, covariances, weights = update_parameters(X, responsibilities)
        # calculate log-likelihood
        current_ll = log_likelihood(X, means, covariances, weights)
        print(f"Iter {iteration+1}: Log-likelihood = {current_ll:.4f}")
        if abs(current_ll - prev_log_likelihood) < tol:
            print("Convergencia alcanzada.")
            break
        prev_log_likelihood = current_ll

    labels = np.argmax(responsibilities, axis=1)
    return labels, means, covariances, current_ll

def plot_gmm_clusters(X: np.ndarray, responsibilities: np.ndarray, means: np.ndarray, covariances: np.ndarray):
    """
    Plots the GMM clusters and their Gaussian components.

    Parameters
    ----------
    X : np.ndarray
        The input data points.
    responsibilities : np.ndarray
        The responsibility matrix from the GMM.
    means : np.ndarray
        The means of the Gaussian components.
    covariances : np.ndarray
        The covariances of the Gaussian components.
    """
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    labels = np.argmax(responsibilities, axis=1)
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        cluster_points = X[mask]
        plt.scatter(
            cluster_points[:, 0], cluster_points[:, 1],
            c=[color], alpha=0.6,
            edgecolors='black',
            label=f'Componente {label}'
        )

        cov = covariances[label]
        mean = means[label]
        eigenvals, eigenvecs = np.linalg.eigh(cov)

        order = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[order]
        eigenvecs = eigenvecs[:, order]
        angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigenvals) * 2
        ellipse = Ellipse(
            xy=(mean[0], mean[1]),
            width=width,
            height=height,
            angle=angle,
            edgecolor=color,
            facecolor='none',
            linestyle='--',
            linewidth=2,
            alpha=0.7
        )
        ax.add_patch(ellipse)

    plt.scatter(
        means[:, 0], means[:, 1],
        c='black', marker='x',
        s=200, linewidths=3,
        label='Medias'
    )

    plt.title('Clustering con GMM', fontsize=14)
    plt.xlabel('A', fontsize=12)
    plt.ylabel('B', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        frameon=True
    )

    plt.tight_layout()
    plt.show()

def plot_gmm_elbow_from_results(k_range: List[int], log_likelihoods: List[float], best_k: Optional[int] = None):
    """
    Plots the log-likelihood values for different numbers of clusters (K) to help select the optimal K for a Gaussian Mixture Model (GMM) using the elbow method.
    Args:
        k_range (List[int]): List of K values (number of clusters) evaluated.
        log_likelihoods (List[float]): Corresponding log-likelihood values for each K.
        best_k (Optional[int], optional): The optimal K value, if known. If provided, a vertical line will be drawn at this value. Defaults to None.
    Displays:
        A matplotlib plot showing log-likelihood vs. number of clusters, with an optional marker for the optimal K.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, log_likelihoods, 'bo-', label='Log-likelihood')
    
    if best_k is not None:
        plt.axvline(x=best_k, color='r', linestyle='--', label=f'K óptimo = {best_k}')
        plt.legend()

    plt.xlabel('Cantidad de clusters (K)')
    plt.ylabel('Log-likelihood')
    plt.title('Selección de K para GMM')
    plt.grid(True)
    plt.show()

def save_gmm_results(data: np.ndarray, responsibilities: np.ndarray, means: np.ndarray, 
                    covariances: np.ndarray, k: int, log_likelihood: float, 
                    K_values: List[int], log_likelihoods: List[float],
                    output_dir: str = '../results_gmm_vectorized'):
    """
    Save GMM clustering results and parameters to files.
    
    Parameters:
    -----------
    data : array-like
        The preprocessed data used for clustering
    responsibilities : array-like
        The responsibility matrix (soft cluster assignments)
    means : array-like
        The component means
    covariances : array-like
        The component covariances
    k : int
        Number of components used
    log_likelihood : float
        Final log-likelihood
    output_dir : str
        Directory to save the results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'preprocessed_data.npy'), data)
    np.save(os.path.join(output_dir, 'responsibilities.npy'), responsibilities)
    np.save(os.path.join(output_dir, 'means.npy'), means)
    np.save(os.path.join(output_dir, 'covariances.npy'), covariances)
    np.savez(os.path.join(output_dir, 'gmm_elbow_data.npz'),
             K_values=K_values, log_likelihoods=log_likelihoods)
    
    # Get hard cluster assignments for statistics
    labels = np.argmax(responsibilities, axis=1)
    stats = {
        'k': int(k),
        'log_likelihood': float(log_likelihood),
        'n_samples': int(len(data)),
        'n_features': int(data.shape[1]),
        'cluster_sizes': [int(np.sum(labels == i)) for i in range(k)]
    }
    with open(os.path.join(output_dir, 'gmm_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"\nResults saved in {output_dir}/")
    print("Files saved:")
    print("- preprocessed_data.npy: preprocessed dataset")
    print("- responsibilities.npy: soft cluster assignments")
    print("- means.npy: component means")
    print("- covariances.npy: component covariances")
    print("- gmm_stats.json: parameters and statistics")
    print("- gmm_elbow_data.npz: K values and log-likelihoods")

def load_gmm_results(input_dir: str = '../results_gmm_vectorized'):
    """
    Load previously saved GMM clustering results.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the saved results
        
    Returns:
    --------
    data : array-like or None
        The preprocessed data used for clustering
    responsibilities : array-like or None
        The responsibility matrix (soft cluster assignments)
    means : array-like or None
        The component means
    covariances : array-like or None
        The component covariances
    stats : dict or None
        Dictionary containing parameters and statistics
    """
    try:
        data = np.load(os.path.join(input_dir, 'preprocessed_data.npy'))
        responsibilities = np.load(os.path.join(input_dir, 'responsibilities.npy'))
        means = np.load(os.path.join(input_dir, 'means.npy'))
        covariances = np.load(os.path.join(input_dir, 'covariances.npy'))
        with open(os.path.join(input_dir, 'gmm_stats.json'), 'r') as f:
            stats = json.load(f)
        gmm_elbow_data = np.load(os.path.join(input_dir, 'gmm_elbow_data.npz'))
        K_values = gmm_elbow_data['K_values']
        log_likelihoods = gmm_elbow_data['log_likelihoods']
            
        print(f"\nLoaded GMM clustering results from {input_dir}/")
        print(f"Dataset shape: {data.shape}")
        print(f"Number of components (K): {stats['k']}")
        print(f"Final log-likelihood: {stats['log_likelihood']:.2f}")
        print("\nCluster sizes:")
        for i, size in enumerate(stats['cluster_sizes']):
            print(f"Component {i}: {size} points")
        
        return data, responsibilities, means, covariances, stats, K_values, log_likelihoods
    
    except Exception as e:
        print(f"Error loading results: {str(e)}")
        return None, None, None, None, None, None, None

def find_best_k_gmm(X: np.ndarray, k_range: List[int], n_runs: int = 3):
    """
    Find the best K using the elbow method with second derivative approach.
    
    Parameters:
    -----------
    X : array-like
        The data to cluster
    k_range : list of int
        Range of K values to try
    n_runs : int
        Number of runs for each K value
        
    Returns:
    --------
    best_k : int
        The optimal number of components
    best_likelihood : float
        The log-likelihood for the best clustering
    best_responsibilities : array-like
        The responsibilities for the best clustering
    best_means : array-like
        The means for the best clustering
    best_covariances : array-like
        The covariances for the best clustering
    """
    likelihoods = []
    all_results = []
    
    for k in k_range:
        print(f"\nTrying k={k}")
        k_likelihoods = []
        k_results = []
        
        for run in range(n_runs):
            print(f"Run {run + 1}/{n_runs}")
            labels, means, covariances, likelihood = gmm(X, k)
            if not np.isnan(likelihood):
                k_likelihoods.append(likelihood)
                responsibilities = np.zeros((len(X), k))
                responsibilities[np.arange(len(X)), labels] = 1
                k_results.append((responsibilities, means, covariances, likelihood))
            print(f"Run {run + 1}: log-likelihood = {likelihood}")
        
        if k_likelihoods:
            max_idx = np.argmax(k_likelihoods)
            max_likelihood = k_likelihoods[max_idx]
            likelihoods.append(max_likelihood)
            all_results.append(k_results[max_idx])
            print(f"Best log-likelihood for k={k}: {max_likelihood}")
        else:
            print(f"Warning: No valid likelihoods for k={k}")
            likelihoods.append(-np.inf)
            all_results.append(None)
    
    # find optimal K using second derivative (elbow method)
    likelihoods = np.array(likelihoods)
    if len(likelihoods) >= 3:
        first_diff = np.diff(likelihoods)
        second_diff = np.diff(first_diff)
        
        elbow_idx = np.argmax(np.abs(second_diff)) + 2  # +2 cause np.diff reduces length twice
        best_k = k_range[elbow_idx]
        
        best_responsibilities, best_means, best_covariances, best_likelihood = all_results[elbow_idx]
    else:
        print("\nWarning: Not enough points to calculate second derivative.")
        print("Defaulting to K=2")
        best_k = 2
        best_responsibilities, best_means, best_covariances, best_likelihood = all_results[1]  # index 1 corresponds to K=2
    
    print(f"\nBest K found: {best_k}")
    print(f"Log-likelihood for best K: {best_likelihood}")

    # elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, likelihoods, 'bo-', label='Log-likelihood')
    plt.axvline(x=best_k, color='r', linestyle='--', label=f'K óptimo = {best_k}')
    plt.xlabel('Cantidad de clusters (K)')
    plt.ylabel('Log-likelihood')
    plt.title('Selección de K para GMM')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return best_k, best_likelihood, best_responsibilities, best_means, best_covariances, k_range, likelihoods

def run_gmm_clustering():
    """Run the GMM clustering and save results"""
    print("Loading data...")
    try:
        # Set random seed for reproducibility
        np.random.seed(42)
        
        data = load_data('../data/raw/clustering.csv')
        print(f"Original data shape: {data.shape}")
        print(f"Data range: [{np.min(data)}, {np.max(data)}]")
        print(f"Any NaN in data: {np.any(np.isnan(data))}")
        
        data, data_mean, data_std = preprocess_data(data)
        print(f"Processed data shape: {data.shape}")
        
        print("\nFinding best K...")
        k_range = list(range(1, 21))
        best_k, best_likelihood, best_responsibilities, best_means, best_covariances, k_range, log_likelihoods = find_best_k_gmm(data, k_range)
        
        save_gmm_results(
            data=data,
            responsibilities=best_responsibilities,
            means=best_means,
            covariances=best_covariances,
            k=best_k,
            log_likelihood=best_likelihood,
            K_values=k_range,
            log_likelihoods=log_likelihoods
        )
        
        plot_gmm_clusters(data, best_responsibilities, best_means, best_covariances)
        
        print(f"\nFinal log-likelihood: {best_likelihood}")
        if not np.isnan(best_likelihood):
            print("Clustering completed successfully!")
        else:
            print("Warning: Final log-likelihood calculation resulted in NaN")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise