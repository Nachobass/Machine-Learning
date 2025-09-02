import numpy as np

def stratified_split_df(df, test_size=0.2, seed=None):
    """
    Splits a DataFrame into stratified training and testing sets based on the target column 'y'.
    This function ensures that the proportion of each class in the target column 'y' is preserved
    in both the training and testing sets.
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data to be split. It must include a column named 'y'
        which represents the target variable.
    test_size : float, optional, default=0.2
        The proportion of the dataset to include in the test split. Must be between 0.0 and 1.0.
    seed : int, optional, default=None
        A random seed for reproducibility. If None, the random number generator is not seeded.
    Returns:
    --------
    tuple of pandas.DataFrame
        A tuple containing two DataFrames:
        - The first DataFrame is the training set.
        - The second DataFrame is the testing set.
    """
    if seed is not None:
        np.random.seed(seed)
    
    test_indices = []
    train_indices = []
    
    for label in df['y'].unique():
        class_indices = df[df['y'] == label].index.to_numpy()
        np.random.shuffle(class_indices)
        split = int(len(class_indices) * (1 - test_size))
        train_indices.extend(class_indices[:split])
        test_indices.extend(class_indices[split:])
    
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    return df.loc[train_indices].reset_index(drop=True), df.loc[test_indices].reset_index(drop=True)