import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from typing import List
import json
import os


def load_data(file_path: str):
    """Load the dataset from CSV file."""
    return np.genfromtxt(file_path, delimiter=',')

def initialize_centroids(X: np.ndarray, k: int):
    """Initialize k centroids randomly from the data points."""
    n_samples = X.shape[0]
    random_indices = np.random.choice(n_samples, k, replace=False)
    return X[random_indices].copy()

def euclidean_distance(X: np.ndarray, centroids: np.ndarray):
    """Calculate euclidean distance between points and centroids."""
    X_expanded = X[:, np.newaxis, :]
    squared_diff = (X_expanded - centroids) ** 2
    return np.sqrt(np.sum(squared_diff, axis=2)) 

def assign_clusters(X: np.ndarray, centroids: np.ndarray):
    """
    Assigns each data point in X to the nearest centroid and computes the total distance.
    Parameters:
        X (np.ndarray): Data points array of shape (n_samples, n_features).
        centroids (np.ndarray): Centroids array of shape (n_clusters, n_features).
    Returns:
        labels (np.ndarray): Array of shape (n_samples,) with the index of the nearest centroid for each data point.
        total_distance (float): Sum of the minimum distances from each data point to its assigned centroid.
    Notes:
        Prints a warning and diagnostic information if NaN values are detected in the total distance calculation.
    """
    distances = euclidean_distance(X, centroids)
    labels = np.argmin(distances, axis=1)
    min_distances = np.min(distances, axis=1)
    total_distance = np.sum(min_distances)
    
    if np.isnan(total_distance):
        print("Warning: NaN detected in total distance calculation")
        print(f"Min distance values: {min_distances}")
        print(f"Distance matrix shape: {distances.shape}")
        print(f"Any NaN in distances: {np.any(np.isnan(distances))}")
    
    return labels, total_distance

def update_centroids(X: np.ndarray, labels: np.ndarray, k: int):
    """
    Updates the centroids for k-means clustering based on current cluster assignments.
    Parameters:
        X (np.ndarray): The dataset, where each row is a data point and each column is a feature.
        labels (np.ndarray): Array of cluster assignments for each data point in X.
        k (int): The number of clusters.
    Returns:
        np.ndarray: The updated centroids as a (k, n_features) array.
    Notes:
        If a cluster has no assigned points, its centroid is reinitialized to a random data point from X.
    """
    n_features = X.shape[1]
    centroids = np.zeros((k, n_features))
    
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            centroids[i] = np.mean(cluster_points, axis=0)
        else:
            # if a cluster is empty, reinitialize it to a random point
            centroids[i] = X[np.random.randint(len(X))]
    
    return centroids

def kmeans(X: np.ndarray, k: int, X_mean: np.ndarray, X_std: np.ndarray, max_iters: int = 100, tol: float = 1e-4):
    """
    Implement K-means clustering algorithm.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Training data
    k : int
        Number of clusters
    max_iters : int
        Maximum number of iterations
    tol : float
        Tolerance for convergence
    """

    centroids = initialize_centroids(X, k)
    prev_distance = float('inf')
    labels = None
    
    for iteration in range(max_iters):
        labels, total_distance = assign_clusters(X, centroids)        
        if iteration == 0:
            print(f"Initial total distance: {total_distance}")
        
        # Check for convergence
        if abs(prev_distance - total_distance) < tol:
            break
            
        prev_distance = total_distance
        centroids = update_centroids(X, labels, k)
    
    # Denormalize centroids
    centroids = centroids * X_std + X_mean
    final_labels, final_distance = assign_clusters(X, centroids)
    
    return final_labels, centroids, final_distance

def plot_clusters(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray, title: str = 'K-means Clustering'):
    """
    Plots the clusters, their centroids, and confidence ellipses for each cluster using matplotlib.

    Parameters:
        X (np.ndarray): The data points, shape (n_samples, 2).
        labels (np.ndarray): Cluster labels for each data point, shape (n_samples,).
        centroids (np.ndarray): Coordinates of cluster centroids, shape (n_clusters, 2).
        title (str, optional): Title of the plot. Defaults to 'K-means Clustering'.

    Displays:
        A scatter plot of the data points colored by cluster, centroids marked with 'x', and ellipses representing the spread of each cluster.
    """
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        cluster_points = X[mask]
        plt.scatter(
            cluster_points[:, 0], cluster_points[:, 1],
            c=[color], alpha=0.6,
            edgecolors='black',
            label=f'Cluster {label}'
        )

        if len(cluster_points) >= 2:
            cov = np.cov(cluster_points, rowvar=False)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            order = eigenvals.argsort()[::-1]
            eigenvals = eigenvals[order]
            eigenvecs = eigenvecs[:, order]

            angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigenvals) * 2 

            ellipse = Ellipse(
                xy=(centroids[label][0], centroids[label][1]),  
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
        centroids[:, 0], centroids[:, 1],
        c='black', marker='x',
        s=200, linewidths=3,
        label='Centroides'
    )

    plt.title(title, fontsize=14)
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

def save_kmeans_results(data: np.ndarray, labels: np.ndarray, centroids: np.ndarray, 
                       k: int, final_distance: float, output_dir: str = '../results_kmeans'):
    """
    Saves the results of a K-means clustering run, including data, labels, centroids, and statistics.
    Parameters:
        data (np.ndarray): The preprocessed dataset used for clustering.
        labels (np.ndarray): Cluster assignments for each data point.
        centroids (np.ndarray): Final centroid positions after clustering.
        k (int): Number of clusters used in K-means.
        final_distance (float): Final value of the distance metric (e.g., inertia or within-cluster sum of squares).
        output_dir (str, optional): Directory where results will be saved. Defaults to '../results_kmeans'.
    Saves:
        - preprocessed_data.npy: The preprocessed dataset.
        - cluster_labels.npy: Cluster assignments for each sample.
        - centroids.npy: Final centroid positions.
        - kmeans_stats.json: Parameters and statistics of the clustering run.
    Creates the output directory if it does not exist.
    Prints a summary of saved files and their locations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'preprocessed_data.npy'), data)    
    np.save(os.path.join(output_dir, 'cluster_labels.npy'), labels)    
    np.save(os.path.join(output_dir, 'centroids.npy'), centroids)
    
    stats = {
        'k': int(k),
        'final_distance': float(final_distance),
        'n_samples': int(len(data)),
        'n_features': int(data.shape[1]),
        'cluster_sizes': [int(np.sum(labels == i)) for i in range(k)]
    }
    with open(os.path.join(output_dir, 'kmeans_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"\nResults saved in {output_dir}/")
    print("Files saved:")
    print("- preprocessed_data.npy: preprocessed dataset")
    print("- cluster_labels.npy: cluster assignments")
    print("- centroids.npy: final centroid positions")
    print("- kmeans_stats.json: parameters and statistics")

def load_kmeans_results(input_dir: str = '../results_kmeans'):
    """
    Load previously saved K-means clustering results.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the saved results
        
    Returns:
    --------
    data : array-like or None
        The preprocessed data used for clustering
    labels : array-like or None
        The cluster labels for each data point
    centroids : array-like or None
        The final centroid positions
    stats : dict or None
        Dictionary containing parameters and statistics
    """
    try:
        data = np.load(os.path.join(input_dir, 'preprocessed_data.npy'))        
        labels = np.load(os.path.join(input_dir, 'cluster_labels.npy'))        
        centroids = np.load(os.path.join(input_dir, 'centroids.npy'))        
        with open(os.path.join(input_dir, 'kmeans_stats.json'), 'r') as f:
            stats = json.load(f)
            
        print(f"\nLoaded K-means clustering results from {input_dir}/")
        print(f"Dataset shape: {data.shape}")
        print(f"Number of clusters (K): {stats['k']}")
        print(f"Final distance: {stats['final_distance']:.2f}")
        print("\nCluster sizes:")
        for i, size in enumerate(stats['cluster_sizes']):
            print(f"Cluster {i}: {size} points")
        
        return data, labels, centroids, stats
    
    except Exception as e:
        print(f"Error loading results: {str(e)}")
        return None, None, None, None

def find_best_k(X: np.ndarray, k_range: List[int], n_runs: int = 5, top_n: int = 16):
    """
    Finds the optimal number of clusters (K) for K-means clustering using the elbow method.
    This function runs K-means clustering for each value of K in the specified range, multiple times per K,
    and selects the best run (with the lowest distance) for each K. It then analyzes the curve of distances
    to find the "elbow" point, which is considered the optimal K, based on the second derivative of the distance curve.
    Args:
        X (np.ndarray): The data to cluster, of shape (n_samples, n_features).
        k_range (List[int]): List of K values (number of clusters) to try.
        n_runs (int, optional): Number of times to run K-means for each K to avoid local minima. Default is 5.
        top_n (int, optional): Number of top K candidates to display based on the second derivative. Default is 16.
    Returns:
        best_k (int): The optimal number of clusters found.
        best_distance (float): The sum of distances for the best K.
        best_labels (np.ndarray): Cluster labels for each sample for the best K.
        best_centroids (np.ndarray): Centroids of the clusters for the best K.
        all_results (dict): Dictionary mapping each K to its best (labels, centroids, distance) tuple.
    """

    distances = []
    all_results = {}

    for k in k_range:
        print(f"\nTrying k={k}")
        k_distances = []
        k_results = []
        
        for run in range(n_runs):
            labels, centroids, distance = kmeans(X, k, X_mean=np.mean(X, axis=0), X_std=np.std(X, axis=0))
            if not np.isnan(distance):
                k_distances.append(distance)
                k_results.append((labels, centroids, distance))
            print(f"Run {run + 1}: distance = {distance:.2f}")
        
        if k_distances:
            min_idx = np.argmin(k_distances)
            min_distance = k_distances[min_idx]
            best_for_k = k_results[min_idx]
            distances.append(min_distance)
            all_results[k] = best_for_k
            print(f"Best distance for k={k}: {min_distance:.2f}")
        else:
            print(f"Warning: No valid distances for k={k}")
            distances.append(float('inf'))
            all_results[k] = None

    distances = np.array(distances)
    
    if len(distances) >= 3:
        first_diff = np.diff(distances)
        second_diff = np.diff(first_diff)
        abs_second_diff = np.abs(second_diff)
        
        top_indices = np.argsort(abs_second_diff)[::-1][:top_n]
        print(f"\nTop {top_n} K candidates based on second derivative (may indicate strong elbows):")
        for i, idx in enumerate(top_indices):
            k_candidate = k_range[idx + 2]
            curvature = second_diff[idx]
            print(f"{i+1}. K = {k_candidate}, second derivative = {curvature:.2f}")
        
        elbow_idx = top_indices[0] + 2
        best_k = k_range[elbow_idx]
        best_labels, best_centroids, best_distance = all_results[best_k]
    else:
        print("\nWarning: Not enough points to calculate second derivative.")
        best_k = 2
        best_labels, best_centroids, best_distance = all_results[best_k]

    print(f"\nBest K found: {best_k}")
    print(f"Distance for best K: {best_distance:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, distances, 'bo-', label='Distancia')
    plt.axvline(x=best_k, color='red', linestyle='--', linewidth=2, label=f'Mejor k = {best_k}')
    plt.xlabel('Cantidad de clusters (K)')
    plt.ylabel('Suma de distancias (L)')
    plt.title('Método del codo para el K óptimo')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return best_k, best_distance, best_labels, best_centroids, all_results

def calculate_silhouette_score(data, labels):
    """
    Calculates the silhouette score for a given clustering assignment.
    The silhouette score measures how similar each point is to its own cluster compared to other clusters.
    A higher score indicates that the points are well clustered. The function handles noise points (label -1)
    by assigning them a silhouette score of -1 and excluding them from the final average.
    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_samples, n_features) containing the data points.
    labels : np.ndarray
        Array of shape (n_samples,) containing the cluster labels for each data point.
        Noise points should be labeled as -1.
    Returns
    -------
    float
        The mean silhouette score for all non-noise points. Returns -1 if there is only one cluster or all points are noise.
    """
    if len(np.unique(labels)) <= 1:  # if only one cluster or all noise
        return -1
    
    n_samples = len(data)
    silhouette_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        if labels[i] == -1:  
            silhouette_scores[i] = -1
            continue
            
        # Calculate a (mean distance to points in same cluster)
        same_cluster = data[labels == labels[i]]
        if len(same_cluster) <= 1:  # If point is alone in its cluster
            a = 0
        else:
            a = np.mean(np.linalg.norm(data[i] - same_cluster, axis=1))
        
        # calculate b (mean distance to points in nearest different cluster)
        b = float('inf')
        for label in np.unique(labels):
            if label != labels[i] and label != -1:
                different_cluster = data[labels == label]
                mean_dist = np.mean(np.linalg.norm(data[i] - different_cluster, axis=1))
                b = min(b, mean_dist)
        
        if b == float('inf'):  
            silhouette_scores[i] = 0
        else:
            silhouette_scores[i] = (b - a) / max(a, b)
    
    valid_scores = silhouette_scores[labels != -1]
    return np.mean(valid_scores) if len(valid_scores) > 0 else -1

def find_best_k_silhouette(X: np.ndarray, k_range: List[int], n_runs: int = 5):
    """
    Finds the optimal number of clusters (K) for KMeans clustering using the silhouette score.

    This function runs KMeans clustering for each value of K in the specified range, multiple times per K,
    and selects the clustering with the highest silhouette score for each K. It then determines the best K
    overall based on the maximum silhouette score.

    Parameters
    ----------
    X : np.ndarray
        The input data array of shape (n_samples, n_features).
    k_range : List[int]
        A list of integer values representing the range of K (number of clusters) to try.
    n_runs : int, optional
        The number of times to run KMeans for each K to account for randomness (default is 5).

    Returns
    -------
    best_k : int
        The value of K (number of clusters) that achieved the highest silhouette score.
    best_score : float
        The highest silhouette score achieved among all K and runs.
    best_labels : np.ndarray
        The cluster labels assigned to each sample for the best K.
    best_centroids : np.ndarray
        The centroids of the clusters for the best K.
    all_results : dict
        A dictionary mapping each K to a tuple (labels, centroids, score) for the best run of that K.
    """

    silhouette_scores = []
    all_results = {}

    for k in k_range:
        print(f"\nTrying k={k}")
        k_scores = []
        k_results = []

        for run in range(n_runs):
            labels, centroids, distance = kmeans(X, k, X_mean=np.mean(X, axis=0), X_std=np.std(X, axis=0))
            score = calculate_silhouette_score(X, labels)
            k_scores.append(score)
            k_results.append((labels, centroids, score))
            print(f"Run {run + 1}: silhouette score = {score:.4f}")

        if k_scores:
            max_idx = np.argmax(k_scores)
            max_score = k_scores[max_idx]
            best_for_k = k_results[max_idx]
            silhouette_scores.append(max_score)
            all_results[k] = best_for_k
            print(f"Best silhouette score for k={k}: {max_score:.4f}")
        else:
            print(f"Warning: No valid silhouette scores for k={k}")
            silhouette_scores.append(-1)
            all_results[k] = None

    silhouette_scores = np.array(silhouette_scores)
    best_idx = np.argmax(silhouette_scores)
    best_k = k_range[best_idx]
    best_score = silhouette_scores[best_idx]
    best_labels, best_centroids, _ = all_results[best_k]

    print(f"\nBest K found: {best_k}")
    print(f"Silhouette score for best K: {best_score:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, 'go-', label='Silhouette Score')
    plt.axvline(x=best_k, color='red', linestyle='--', linewidth=2, label=f'Mejor k = {best_k}')
    plt.xlabel('Cantidad de clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Selección de K usando Silhouette Score')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_k, best_score, best_labels, best_centroids, all_results

def preprocess_data(data, plot=False):
    """
    Preprocesses the input data by removing the first column, handling missing values, normalizing features, and optionally plotting the data distribution.
    Steps performed:
    1. Removes the first column from the input data.
    2. Optionally plots the distribution of the remaining data (first two components).
    3. Imputes missing (NaN) values in each column with the column mean.
    4. Normalizes each feature to have zero mean and unit variance.
    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features). The first column is removed during preprocessing.
    plot : bool, optional (default=False)
        If True, plots the distribution of the first two features after removing the first column.
    Returns
    -------
    data_normalized : np.ndarray
        The preprocessed and normalized data array of shape (n_samples, n_features - 1).
    data_mean : np.ndarray
        The mean of each feature (after removing the first column), used for normalization.
    data_std : np.ndarray
        The standard deviation of each feature (after removing the first column), used for normalization.
    """

    data = data[:, 1:]
    if plot:
        plt.figure(figsize=(8, 6)) 

        plt.scatter(
            data[:, 0], data[:, 1],
            c='cornflowerblue', 
            edgecolor='k',        
            alpha=0.7,          
            s=60                      
        )

        plt.title("Distribución de los datos", fontsize=14)
        plt.xlabel("Componente 1", fontsize=12)
        plt.ylabel("Componente 2", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)  
        plt.tight_layout()
        plt.show()
    
    # handle NaNs values by imputing with means
    nan_mask = np.isnan(data)
    nan_counts = np.sum(nan_mask, axis=0)
    
    print("\nStatistics of NaN values:")
    for i, count in enumerate(nan_counts):
        print(f"Column {i}: {count} NaN values")
    print(f"Total NaN values in dataset: {np.sum(nan_counts)}")
    
    col_means = np.nanmean(data, axis=0)
    for i in range(data.shape[1]):
        mask = nan_mask[:, i]
        data[mask, i] = col_means[i]
    
    print(f"\nAny NaN values after imputation?: {np.any(np.isnan(data))}")

    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    data_std[data_std == 0] = 1 
    data_normalized = (data - data_mean) / data_std

    return data_normalized, data_mean, data_std

def run_kmeans_clustering(silhouette: bool = False):
    """
    Runs the K-Means clustering pipeline on a dataset, including data loading, preprocessing,
    optimal K selection, clustering, result saving, and visualization.
    Parameters:
        silhouette (bool, optional): If True, selects the optimal number of clusters using the silhouette score.
                                     If False, uses the sum of distances to select the best K. Default is False.
    Workflow:
        1. Loads and preprocesses the data from a CSV file.
        2. Determines the optimal number of clusters (K) using either silhouette score or sum of distances.
        3. Performs K-Means clustering with the selected K.
        4. Saves clustering results and plots the clustered data.
        5. Prints diagnostic information and handles exceptions.
    Raises:
        Exception: Propagates any exception that occurs during the clustering process.
    """

    print("Loading data...")
    try:
        # Load and preprocess data
        data = load_data('../data/raw/clustering.csv')
        print(f"Original data shape: {data.shape}")
        print(f"Data range: [{np.min(data)}, {np.max(data)}]")
        print(f"Any NaN in data: {np.any(np.isnan(data))}")
        
        data, data_mean, data_std = preprocess_data(data, plot=True)
        print(f"Processed data shape: {data.shape}")
        print(f"Processed data range: [{np.min(data)}, {np.max(data)}]")
        
        print("\nFinding best K...")
        k_range = list(range(1, 21))

        if silhouette:
            best_k, best_score, best_labels, best_centroids, all_results = find_best_k_silhouette(data, k_range)
        else:
            best_k, best_distance, best_labels, best_centroids, all_results = find_best_k(data, k_range)

        selected_k = best_k     # clusters q quiero plotear
        selected_labels, selected_centroids, selected_distance = all_results[selected_k]
        
        if silhouette:
            save_kmeans_results(
                data=data,
                labels=selected_labels,
                centroids=selected_centroids,
                k=selected_k,
                final_distance=selected_distance,
                output_dir='../results_kmeans_silhouette'
            )
        else:
            save_kmeans_results(
                data=data,
                labels=selected_labels,
                centroids=selected_centroids,
                k=selected_k,
                final_distance=selected_distance
            )
        
        plot_clusters(data, selected_labels, selected_centroids)
        
        print(f"\nFinal sum of distances: {selected_distance:.2f}")
        if not np.isnan(selected_distance):
            print("Clustering completed successfully!")
        else:
            print("Warning: Final distance calculation resulted in NaN")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise