import numpy as np


def train_val_split_df(df, val_size=0.2, seed=None):
    """
    Splits a DataFrame into training and validation sets.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame to be split.
    val_size : float, optional, default=0.2
        The proportion of the data to include in the validation set. 
        Must be between 0 and 1.
    seed : int or None, optional, default=None
        The random seed for reproducibility. If None, the random 
        state is not set.

    Returns:
    --------
    train : pandas.DataFrame
        The training set DataFrame.
    val : pandas.DataFrame
        The validation set DataFrame.
    """
    if seed is not None:
        np.random.seed(seed)
    train = df.sample(frac=1-val_size, random_state=seed)
    val = df.drop(train.index)
    return train, val


def cross_val(X, y, model_class, lambdas, k=5):
    """
    Perform k-fold cross-validation to select the best regularization parameter (lambda)
    for a given model class using mean squared error (MSE) as the evaluation metric.
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : numpy.ndarray
        Target vector of shape (n_samples).
    model_class : class
        A class representing the model to be trained. The class must implement
        a constructor that accepts X and y, a `train_pseudo_inverse_regularized` method
        that accepts an `l2` parameter for regularization, and a `predict` method.
    lambdas : list or numpy.ndarray
        A list or array of regularization parameters (lambda values) to evaluate.
    k : int, optional
        The number of folds for cross-validation. Default is 5.
    Returns:
    --------
    best_lambda : float
        The regularization parameter (lambda) that results in the lowest average MSE
        across the k folds.
    mse_values : list
        A list of average MSE values for each lambda in the `lambdas` list.
    Notes:
    ------
    - The data is shuffled before splitting into folds to ensure randomness.
    - The function assumes that the model class provided has the required methods
      for training and prediction.
    """
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    fold_size = n // k  # Size of each fold
    mse_values = []
    
    best_lambda = None
    best_mse = np.inf
    
    for lamb in lambdas:
        mse_folds = []
        
        for i in range(k):
            # Create the validation and training sets for this fold
            val_idx = indices[i * fold_size : (i + 1) * fold_size]
            train_idx = np.setdiff1d(indices, val_idx)
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            model = model_class(X_train, y_train)
            model.train_pseudo_inverse_regularized(l2=lamb)
            
            y_pred = model.predict(X_val)
            
            mse_fold = np.mean((y_val - y_pred) ** 2)
            mse_folds.append(mse_fold)
        
        # Mean MSE across the k folds
        mse_avg = np.mean(mse_folds)
        mse_values.append(mse_avg)
        
        if mse_avg < best_mse:
            best_mse = mse_avg
            best_lambda = lamb
    
    return best_lambda, mse_values