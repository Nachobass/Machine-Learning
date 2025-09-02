import matplotlib.pyplot as plt
import numpy as np

def plot_reconstructions(X_original, X_reconstructed, n=5, labels=None, title=None, title_size=14):
    """
    Plots original and reconstructed images side by side for visual comparison.

    Args:
        X_original (np.ndarray): Array of original images, each flattened (e.g., shape (n_samples, 784)).
        X_reconstructed (np.ndarray): Array of reconstructed images, each flattened (same shape as X_original).
        n (int, optional): Number of images to display. Defaults to 5.
        labels (list or np.ndarray, optional): Labels corresponding to the images. If provided, labels are shown above original images.
        title (str, optional): Title for the entire plot. Defaults to None.
        title_size (int, optional): Font size for the plot title. Defaults to 14.

    Returns:
        None. Displays a matplotlib figure with original and reconstructed images.
    """
    plt.figure(figsize=(2*n, 4))
    for i in range(n):
        # original image
        plt.subplot(2, n, i + 1)
        plt.imshow(X_original[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        original_title = "Original"
        if labels is not None:
            original_title += f"\nLabel: {labels[i]}"
        plt.title(original_title)

        # reconstructed image
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(X_reconstructed[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title("Reconstruida")

    if title:
        plt.suptitle(title, fontsize=title_size, y=1.05)
    plt.tight_layout()
    plt.show()

def stratified_train_val_split(X, y, train_ratio=0.8, seed=42):
    """
    Splits the dataset into stratified training and validation sets.

    This function ensures that the class distribution in both the training and validation sets
    matches the distribution in the original dataset.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target array of shape (n_samples,).
    train_ratio : float, optional (default=0.8)
        Proportion of the dataset to include in the training split.
    seed : int, optional (default=42)
        Random seed for reproducibility.

    Returns
    -------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training target array.
    X_val : np.ndarray
        Validation feature matrix.
    y_val : np.ndarray
        Validation target array.
    """
    np.random.seed(seed)
    train_indices = []
    val_indices = []

    classes = np.unique(y)
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)
        n_train = int(train_ratio * len(cls_indices))
        train_indices.extend(cls_indices[:n_train])
        val_indices.extend(cls_indices[n_train:])

    # Shuffle the final indices
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    return X[train_indices], y[train_indices], X[val_indices], y[val_indices]