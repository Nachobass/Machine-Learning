import numpy as np
import pandas as pd


def handle_missing_values(df, age_mean, age_std, rooms_mean, rooms_std, seed=None):
    """
    Handles missing values in the 'age' and 'rooms' columns of a DataFrame by imputing
    them with random values drawn from a normal distribution.
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data with missing values.
    age_mean : float
        The mean value to use for imputing missing values in the 'age' column.
    age_std : float
        The standard deviation to use for imputing missing values in the 'age' column.
    rooms_mean : float
        The mean value to use for imputing missing values in the 'rooms' column.
    rooms_std : float
        The standard deviation to use for imputing missing values in the 'rooms' column.
    seed : int, optional
        A seed value for the random number generator to ensure reproducibility. Default is None.
    Returns:
    --------
    None
        The function modifies the input DataFrame in place, filling missing values in the
        'age' and 'rooms' columns with randomly generated values.
    """
    if seed is not None:
        np.random.seed(seed)
    
    age_missing_mask = df['age'].isna()
    rooms_missing_mask = df['rooms'].isna()

    df.loc[age_missing_mask, 'age'] = np.random.normal(age_mean, age_std, age_missing_mask.sum()).round()
    df.loc[rooms_missing_mask, 'rooms'] = np.random.normal(rooms_mean, rooms_std, rooms_missing_mask.sum()).round()


def normalize_data(df, mean, std, columns_to_not_normalize):
    """
    Normalizes the continuous columns of a DataFrame using the provided mean and standard deviation,
    while excluding specified columns from normalization.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be normalized.
        mean (pd.Series or pd.DataFrame): The mean values for normalization, corresponding to the columns in `df`.
        std (pd.Series or pd.DataFrame): The standard deviation values for normalization, corresponding to the columns in `df`.
        columns_to_not_normalize (list): A list of column names to exclude from normalization.

    Returns:
        pd.DataFrame: A DataFrame with the normalized continuous columns and the excluded columns unchanged.
    """
    df_continuas = df.drop(columns=["area_units"] + columns_to_not_normalize)
    df_normalized = (df_continuas - mean) / std
    df_normalized = pd.concat([df_normalized, df[columns_to_not_normalize]], axis=1)

    return df_normalized


def desnormalize_data(y_pred, train_df, target):
    """
    Denormalizes predicted data using the mean and standard deviation of the target column 
    from the training dataset.
    Parameters:
        y_pred (array-like): The normalized predicted values to be denormalized.
        train_df (pd.DataFrame): The training dataset containing the target column.
        target (str): The name of the target column in the training dataset.
    Returns:
        array-like: The denormalized predicted values.
    """
    target_mean = train_df[target].mean()
    target_std = train_df[target].std()
    
    return (y_pred * target_std) + target_mean