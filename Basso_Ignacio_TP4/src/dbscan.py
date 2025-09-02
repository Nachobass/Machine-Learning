import numpy as np
import matplotlib.pyplot as plt
import json
import os
from collections import deque
from kmeans import load_data, preprocess_data, calculate_silhouette_score


def save_results(data, labels, eps, min_points, output_dir='../results'):
    """
    Save clustering results and parameters to files.
    
    Parameters:
    -----------
    data : array-like
        The preprocessed data used for clustering
    labels : array-like
        The cluster labels for each data point
    eps : float
        The epsilon parameter used
    min_points : int
        The min_points parameter used
    output_dir : str
        Directory to save the results
    """
    os.makedirs(output_dir, exist_ok=True)    
    np.save(os.path.join(output_dir, 'preprocessed_data.npy'), data)    
    np.save(os.path.join(output_dir, 'cluster_labels.npy'), labels)
    
    stats = {
        'eps': eps,
        'min_points': min_points,
        'n_clusters': len(np.unique(labels[labels != -1])),
        'n_noise': int(np.sum(labels == -1)),
        'n_samples': len(data),
        'n_features': data.shape[1]
    }
    with open(os.path.join(output_dir, 'clustering_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"\nResults saved in {output_dir}/")
    print("Files saved:")
    print("- preprocessed_data.npy: preprocessed dataset")
    print("- cluster_labels.npy: cluster assignments")
    print("- clustering_stats.json: parameters and statistics")

def load_results(input_dir='../results'):
    """
    Load previously saved clustering results.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the saved results
        
    Returns:
    --------
    data : array-like
        The preprocessed data used for clustering
    labels : array-like
        The cluster labels for each data point
    stats : dict
        Dictionary containing parameters and statistics
    """
    try:
        data = np.load(os.path.join(input_dir, 'preprocessed_data.npy'))        
        labels = np.load(os.path.join(input_dir, 'cluster_labels.npy'))        
        with open(os.path.join(input_dir, 'clustering_stats.json'), 'r') as f:
            stats = json.load(f)
            
        print(f"\nLoaded clustering results from {input_dir}/")
        print(f"Dataset shape: {data.shape}")
        print(f"Number of clusters: {stats['n_clusters']}")
        print(f"Number of noise points: {stats['n_noise']}")
        
        return data, labels, stats
    
    except Exception as e:
        print(f"Error loading results: {str(e)}")
        return None, None, None

class DBSCAN:
    def __init__(self, eps, min_points):
        """
        Initializes the DBSCAN clustering algorithm with the specified parameters.

        Parameters:
            eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
            min_points (int): The minimum number of points required to form a dense region (core point).
        """
        self.eps = eps
        self.min_points = min_points

    def _region_query(self, data, idx, tree=None):
        """
        Finds the indices of all points in the dataset within a given radius (eps) of the point at index `idx`.

        If a spatial tree is provided, it uses the tree's efficient query method; otherwise, it computes distances directly.

        Parameters:
            data (np.ndarray): The dataset as a 2D NumPy array of shape (n_samples, n_features).
            idx (int): The index of the point to query neighbors for.
            tree (object, optional): A spatial tree object (e.g., KDTree or BallTree) with a `query_ball_point` method. Defaults to None.

        Returns:
            np.ndarray: Indices of points within `eps` distance of the point at `idx`.
        """
        if tree is not None:
            return tree.query_ball_point(data[idx], r=self.eps)
        else:
            dists = np.linalg.norm(data - data[idx], axis=1)
            return np.where(dists <= self.eps)[0]

    def fit(self, data):
        """
        Perform DBSCAN clustering on the input data.

        Parameters
        ----------
        data : np.ndarray
            The input data to cluster, expected to be a 2D array of shape (n_samples, n_features).

        Returns
        -------
        labels : np.ndarray
            Array of shape (n_samples,) containing cluster labels for each point.
            Noise points are labeled as -1.
        """
        n = data.shape[0]
        labels = np.full(n, -1)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(data)
        except ImportError:
            tree = None

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True

            neighbors = self._region_query(data, i, tree)
            if len(neighbors) < self.min_points:
                continue

            labels[i] = cluster_id
            queue = deque(neighbors)

            while queue:
                j = queue.popleft()
                if not visited[j]:
                    visited[j] = True
                    new_neighbors = self._region_query(data, j, tree)
                    if len(new_neighbors) >= self.min_points:
                        queue.extend(n for n in new_neighbors if labels[n] == -1)

                if labels[j] == -1:
                    labels[j] = cluster_id

            cluster_id += 1

        return labels


def find_best_parameters(data, eps_range=None, minpts_range=None):
    """
    Finds the best parameters (epsilon and minPts) for DBSCAN clustering on the given dataset.
    This function searches over specified or automatically determined ranges of epsilon (eps) and minimum points (minPts)
    to identify the parameter combination that yields the best clustering performance according to a custom scoring metric.
    The metric balances silhouette score, noise ratio, and the number of clusters.
    Parameters:
        data (np.ndarray): The input data array of shape (n_samples, n_features).
        eps_range (array-like, optional): Range of epsilon values to try. If None, a range is estimated from the data.
        minpts_range (array-like, optional): Range of minPts values to try. If None, a range is set based on data dimensionality.
    Returns:
        tuple: (best_eps, best_minpts)
            best_eps (float): The epsilon value that resulted in the best clustering.
            best_minpts (int): The minPts value that resulted in the best clustering.
    """

    if eps_range is None:
        # calculate a reasonable range for epsilon based on data distribution
        distances = []
        sample_size = min(1000, len(data))  
        indices = np.random.choice(len(data), sample_size, replace=False)
        for i in indices:
            dist = np.linalg.norm(data - data[i], axis=1)
            distances.extend(dist)
        
        percentiles = [1, 5, 10, 15, 20, 25, 30, 35, 40]
        eps_range = np.unique(np.percentile(distances, percentiles))
        eps_range = np.append(eps_range, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        eps_range = np.unique(eps_range)
        eps_range.sort()
    
    if minpts_range is None:
        dims = data.shape[1]
        minpts_range = list(range(dims + 1, dims + 16, 2)) 
    
    best_score = float('-inf')
    best_eps = eps_range[0]  # Default to first epsilon if nothing better found
    best_minpts = minpts_range[0]  # Default to first minpts if nothing better found
    best_n_clusters = 0
    results = []
    
    print("\nSearching parameters:")
    print(f"Epsilon range: {eps_range}")
    print(f"MinPts range: {minpts_range}")
    
    for eps in eps_range:
        for minpts in minpts_range:
            dbscan = DBSCAN(eps=eps, min_points=minpts)
            labels = dbscan.fit(data)
            
            n_clusters = len(np.unique(labels[labels != -1]))
            n_noise = np.sum(labels == -1)
            noise_ratio = n_noise / len(data)
            
            # skip if too many clusters or too much noise
            if n_clusters < 2 or noise_ratio > 0.8: 
                continue
            
            silhouette = calculate_silhouette_score(data, labels)
            
            # modified scoring metric to balance between cluster quality and noise
            score = silhouette * (1 - noise_ratio) * np.log2(n_clusters + 1)
            results.append({
                'eps': eps,
                'minpts': minpts,
                'n_clusters': n_clusters,
                'noise_ratio': noise_ratio,
                'silhouette': silhouette,
                'score': score
            })
            if score > best_score:
                best_score = score
                best_eps = eps
                best_minpts = minpts
                best_n_clusters = n_clusters
    
    if len(results) == 0:
        print("\nWarning: No valid parameter combinations found with current criteria.")
        print("Using default parameters.")
        return eps_range[len(eps_range)//2], minpts_range[0]  
    
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print("\nTop 3 parameter combinations:")
    for i, result in enumerate(results[:3]):
        print(f"\n{i+1}. Configuration:")
        print(f"   eps = {result['eps']}")
        print(f"   minPts = {result['minpts']}")
        print(f"   Clusters: {result['n_clusters']}")
        print(f"   Noise ratio: {result['noise_ratio']:.2%}")
        print(f"   Silhouette score: {result['silhouette']:.3f}")
        print(f"   Overall score: {result['score']:.3f}")
    
    print("\nBest parameters found:")
    print(f"eps = {best_eps}")
    print(f"minPts = {best_minpts}")
    print(f"Number of clusters: {best_n_clusters}")
    print(f"Score: {best_score}")
    
    return best_eps, best_minpts

def plot_clusters_dbscan(data, labels, title):
    """Plot DBSCAN clusters with dashed circles and noise in black using many distinct colors."""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])
    colormap = plt.cm.get_cmap('nipy_spectral', n_clusters)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Plot noise points (label == -1)
    noise_mask = labels == -1
    plt.scatter(data[noise_mask, 0], data[noise_mask, 1],
                c='black', label='Ruido', alpha=0.5)

    # Plot each cluster
    for idx, label in enumerate(unique_labels):
        if label == -1:
            continue  # Skip noise

        mask = labels == label
        cluster_points = data[mask]
        color = colormap(idx / n_clusters)

        # Scatter plot del cluster
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    c=[color], alpha=0.6,
                    edgecolors='black', label=f'Cluster {label}')

    plt.title(title, fontsize=14)
    plt.xlabel('A', fontsize=12)
    plt.ylabel('B', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Leyenda afuera
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)

    plt.tight_layout()
    plt.show()

def run_dbscan_clustering():
    """
    Runs the complete DBSCAN clustering pipeline on a dataset.
    This function performs the following steps:
    1. Loads the data from a CSV file.
    2. Prints basic statistics about the data (shape, range, presence of NaNs).
    3. Preprocesses the data (e.g., normalization or standardization).
    4. Finds the best DBSCAN parameters (epsilon and minimum points).
    5. Runs the DBSCAN clustering algorithm with the best parameters.
    6. Saves the clustering results.
    7. Prints the number of clusters and noise points found.
    8. Plots the resulting clusters.
    Raises:
        Exception: If any error occurs during the process, it is printed and re-raised.
    """
    print("Loading data...")
    try:
        
        data = load_data('../data/raw/clustering.csv')
        print(f"Original data shape: {data.shape}")
        print(f"Data range: [{np.min(data)}, {np.max(data)}]")
        print(f"Any NaN in data: {np.any(np.isnan(data))}")
        
        data, data_mean, data_std = preprocess_data(data)
        print(f"Processed data shape: {data.shape}")
        
        print("\nFinding best parameters...")
        best_eps, best_minpts = find_best_parameters(data)

        print("\nRunning DBSCAN with best parameters...")
        dbscan = DBSCAN(eps=best_eps, min_points=best_minpts)
        labels = dbscan.fit(data)

        save_results(data, labels, best_eps, best_minpts)

        n_clusters = len(np.unique(labels[labels != -1]))
        n_noise = np.sum(labels == -1)
        print(f"\nClustering Results:")
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise} ({n_noise/len(data):.1%} of data)")

        plot_clusters_dbscan(data, labels, f'DBSCAN Clustering (eps={best_eps:.2f}, minPts={best_minpts})')
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise