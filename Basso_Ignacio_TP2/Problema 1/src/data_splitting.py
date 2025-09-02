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