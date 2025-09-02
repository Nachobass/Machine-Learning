import numpy as np

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
        Target vector of shape (n_samples,).
    train_ratio : float, optional (default=0.8)
        Proportion of the dataset to include in the training split.
    seed : int, optional (default=42)
        Random seed for reproducibility.

    Returns
    -------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training target vector.
    X_val : np.ndarray
        Validation feature matrix.
    y_val : np.ndarray
        Validation target vector.
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

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    return X[train_indices], y[train_indices], X[val_indices], y[val_indices]
