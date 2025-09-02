import numpy as np


def prepare_data(df, features, target, add_bias=True):
    """
    Prepares the data for machine learning by extracting features and target variables
    from a DataFrame and optionally adding a bias term to the features.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        features (list of str): A list of column names to be used as features.
        target (str): The name of the column to be used as the target variable.
        add_bias (bool, optional): Whether to add a bias term (a column of ones) 
            to the features. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): The feature matrix, with shape (n_samples, n_features).
              If `add_bias` is True, an additional column of ones is added.
            - y (numpy.ndarray): The target variable array, with shape (n_samples,).
    """
    X = df[features].values
    if add_bias:
        X = np.c_[np.ones((X.shape[0], 1)), X]
    y = df[target].values
    return X, y