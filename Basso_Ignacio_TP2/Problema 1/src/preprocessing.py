import numpy as np
import pandas as pd

TARGET = 'Diagnosis'

def prepare_data(df, features, target):
    """
    Prepares the data for machine learning by extracting features and target variables
    and adding an intercept term to the feature matrix.

    Args:
        df (pandas.DataFrame): The input dataframe containing the data.
        features (list of str): A list of column names to be used as features.
        target (str): The name of the column to be used as the target variable.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): The feature matrix with an added intercept term.
            - y (numpy.ndarray): The target variable array.
    """
    X = df[features].values
    X = np.c_[np.ones((X.shape[0], 1)), X] # Add intercept term
    y = df[target].values
    return X,y


def hamming_distance(x, y):
    """
    Calculate the Hamming distance between two sequences.

    The Hamming distance is defined as the proportion of positions at which 
    the corresponding elements of two sequences are different.

    Parameters:
    x (array-like): The first sequence.
    y (array-like): The second sequence. Must have the same length as `x`.

    Returns:
    float: The Hamming distance, which is the number of differing positions 
           divided by the total length of the sequences.
    """
    return np.sum(x != y) / len(x)


def knn_impute_with_categorical(df_train, df_val=None, base_k=15, lambda_hamming=1.0):
    """
    Imputes missing values in a DataFrame using a K-Nearest Neighbors (KNN) approach 
    that considers both numerical and categorical features.

    Parameters:
    -----------
    df_train : pandas.DataFrame
        The training DataFrame containing missing values to be imputed.
    df_val : pandas.DataFrame, optional
        The validation DataFrame to apply the same imputation procedure. If None, 
        only the training DataFrame is processed. Default is None.
    base_k : int, optional
        The base number of neighbors to consider for KNN imputation. The actual 
        number of neighbors is dynamically adjusted based on the size of the data. 
        Default is 15.
    lambda_hamming : float, optional
        The weight applied to the Hamming distance when combining it with the 
        Euclidean distance for mixed-type data. Default is 1.0.

    Returns:
    --------
    pandas.DataFrame or tuple of pandas.DataFrame
        If `df_val` is None, returns the imputed training DataFrame. Otherwise, 
        returns a tuple containing the imputed training and validation DataFrames.

    Notes:
    ------
    - Numerical features are normalized before calculating Euclidean distances.
    - Categorical features are converted to numeric codes before calculating 
      Hamming distances.
    - Missing values in numerical columns are initially imputed with the mean, 
      and missing values in categorical columns are initially imputed with the mode.
    - The number of neighbors `k` is dynamically adjusted to be at least 3 and 
      at most `base_k`, depending on the size of the available data.
    - For numerical columns, missing values are imputed with the mean of the 
      nearest neighbors. For categorical columns, missing values are imputed 
      with the mode of the nearest neighbors.
    """
    df_train_original = df_train.copy()
    df_train_imputed = df_train.copy()

    # Impute initial values ​​with mean (numeric) and mode (categorical)
    for col in df_train.columns:
        if df_train[col].dtype in [np.float64, np.int64]:
            df_train_imputed.fillna({col: df_train[col].mean()}, inplace=True)
        else:
            df_train_imputed.fillna({col: df_train[col].mode()[0]}, inplace=True)

    numeric_columns = df_train.select_dtypes(include=[np.number]).columns
    categorical_columns = df_train.select_dtypes(exclude=[np.number]).columns

    df_train_numeric = df_train_imputed[numeric_columns]
    df_train_numeric = (df_train_numeric - df_train_numeric.mean()) / df_train_numeric.std()

    # Convert categories to numeric codes
    df_train_categorical = df_train_imputed[categorical_columns].apply(lambda x: x.astype('category').cat.codes)

    # Imputation by KNN
    for col in df_train.columns:
        nan_rows = df_train_original[col].isna()

        if nan_rows.sum() == 0:
            continue  

        known_data_num = df_train_numeric.loc[~nan_rows]  
        known_data_cat = df_train_categorical.loc[~nan_rows]  

        missing_data_num = df_train_numeric.loc[nan_rows]
        missing_data_cat = df_train_categorical.loc[nan_rows]

        for idx in missing_data_num.index:
            # Euclidean distance
            distances_num = np.linalg.norm(known_data_num - missing_data_num.loc[idx], axis=1)

            # Hamming distance
            distances_cat = (known_data_cat != missing_data_cat.loc[idx]).sum(axis=1)

            # Total distance
            distances = np.sqrt(distances_num) + lambda_hamming * distances_cat

            # Dynamic selection of k (minimum 3, maximum base_k)
            k = min( base_k, max( 3, int(np.sqrt( len(known_data_num) )) ) )

            # Nearest neighbors
            nearest_indices = known_data_num.index[np.argsort(distances)[:k]]

            if df_train[col].dtype in [np.float64, np.int64]:  
                df_train_original.at[idx, col] = df_train.loc[nearest_indices, col].mean()
            else:  
                df_train_original.at[idx, col] = df_train.loc[nearest_indices, col].mode()[0]

    if df_val is None:
        return df_train_original

    # Apply the same procedure to the validation set
    df_val_original = df_val.copy()
    df_val_imputed = df_val.copy()

    for col in df_val.columns:
        if df_val[col].dtype in [np.float64, np.int64]:
            df_val_imputed.fillna({col: df_train[col].mean()}, inplace=True)
        else:
            df_val_imputed.fillna({col: df_train[col].mode()[0]}, inplace=True)

    df_val_numeric = df_val_imputed[numeric_columns]
    df_val_numeric = (df_val_numeric - df_train_numeric.mean()) / df_train_numeric.std()
    df_val_categorical = df_val_imputed[categorical_columns].apply(lambda x: x.astype('category').cat.codes)

    for col in df_val.columns:
        nan_rows = df_val_original[col].isna()

        if nan_rows.sum() == 0:
            continue  

        missing_data_num = df_val_numeric.loc[nan_rows]
        missing_data_cat = df_val_categorical.loc[nan_rows]

        for idx in missing_data_num.index:
            distances_num = np.linalg.norm(known_data_num - missing_data_num.loc[idx], axis=1)
            distances_cat = (known_data_cat != missing_data_cat.loc[idx]).sum(axis=1)
            distances = np.sqrt(distances_num) + lambda_hamming * distances_cat

            k = min(base_k, max(3, int(np.sqrt(len(known_data_num)))))

            nearest_indices = known_data_num.index[np.argsort(distances)[:k]]

            if df_val[col].dtype in [np.float64, np.int64]:  
                df_val_original.at[idx, col] = df_train.loc[nearest_indices, col].mean()
            else:  
                df_val_original.at[idx, col] = df_train.loc[nearest_indices, col].mode()[0]

    return df_train_original, df_val_original


def one_hot_encode(df, categorical_columns):
    """
    Perform one-hot encoding on the specified categorical columns of a DataFrame.

    This function creates binary indicator variables for each category in the 
    specified categorical columns. If a column contains boolean values, they 
    are first converted to integers before encoding. For categorical columns 
    with only one unique value, all categories are retained during encoding. 
    For other categorical columns, the first category is dropped to avoid 
    multicollinearity.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data to be encoded.
    categorical_columns : list of str
        A list of column names in the DataFrame that should be one-hot encoded.

    Returns:
    --------
    pandas.DataFrame
        A new DataFrame with the specified categorical columns one-hot encoded.
        The original DataFrame remains unchanged.

    Notes:
    ------
    - Columns with boolean data types are converted to integers before encoding.
    - For columns with only one unique value, no category is dropped during encoding.
    - For columns with multiple unique values, the first category is dropped 
      to avoid multicollinearity.
    """
    df_encoded = df.copy()

    # Convert booleans to integers before one-hot encoding
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'bool':
            df_encoded[col] = df_encoded[col].astype(int)

    for col in categorical_columns:
        unique_vals = df_encoded[col].nunique()
        if unique_vals == 1:
            df_encoded = pd.get_dummies(df_encoded, columns=[col], drop_first=False)
        else:
            df_encoded = pd.get_dummies(df_encoded, columns=[col], drop_first=True)

    return df_encoded


def correct_out_of_range_values(df, feature_ranges):
    """
    Corrects out-of-range values in a DataFrame by replacing them with NaN.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data to be processed.
    feature_ranges : dict
        A dictionary where keys are feature (column) names and values are tuples
        specifying the minimum and maximum allowable values for the corresponding feature.
        Example: {'feature1': (min_val1, max_val1), 'feature2': (min_val2, max_val2)}

    Returns:
    --------
    pandas.DataFrame
        A copy of the input DataFrame with out-of-range values replaced by NaN.
    """
    df_corrected = df.copy()

     # Replace out-of-range values ​​with NaN
    for feature, (min_val, max_val) in feature_ranges.items():
        if feature in df_corrected.columns:
            out_of_range = (df_corrected[feature] < min_val) | (df_corrected[feature] > max_val)
            df_corrected.loc[out_of_range, feature] = np.nan

    return df_corrected


def normalize_df(df, train_mean=None, train_std=None, train=True, exclude_columns=None):
    """
    Normalizes the columns of a DataFrame by subtracting the mean and dividing by the standard deviation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to normalize.
    train_mean : pandas.Series or None, optional
        The mean values of the columns to normalize, used when `train` is False. Default is None.
    train_std : pandas.Series or None, optional
        The standard deviation values of the columns to normalize, used when `train` is False. Default is None.
    train : bool, optional
        If True, the function calculates the mean and standard deviation from the given DataFrame.
        If False, the function uses the provided `train_mean` and `train_std` for normalization. Default is True.
    exclude_columns : list or None, optional
        A list of column names to exclude from normalization. Default is None.

    Returns:
    --------
    pandas.DataFrame
        A new DataFrame with the normalized columns.
    pandas.Series (only if `train` is True)
        The mean values of the normalized columns.
    pandas.Series (only if `train` is True)
        The standard deviation values of the normalized columns.

    Raises:
    -------
    ValueError
        If `train` is False and either `train_mean` or `train_std` is not provided.

    Notes:
    ------
    - Columns with a standard deviation of 0 are not normalized to avoid division by zero.
    - The function creates a copy of the input DataFrame to avoid modifying the original data.
    """
    if exclude_columns is None:
        exclude_columns = []

    df_copy = df.copy()
    columns_to_normalize = [col for col in df.columns if col not in exclude_columns]

    if train:
        mean = df_copy[columns_to_normalize].mean()
        std = df_copy[columns_to_normalize].std()
        std[std == 0] = 1  # Avoid division by zero
        df_copy[columns_to_normalize] = (df_copy[columns_to_normalize] - mean) / std
        return df_copy, mean, std
    else:
        if train_mean is None or train_std is None:
            raise ValueError("train_mean y train_std deben ser proporcionados si train es False.")
        df_copy[columns_to_normalize] = (df_copy[columns_to_normalize] - train_mean) / train_std
        return df_copy


def process_data(train_df, categorical_columns, val_df=None, feature_ranges=None, k=15, target=TARGET):
    """
    Processes the input data by handling missing values, encoding categorical features, 
    and normalizing numerical features.

    Args:
        train_df (pd.DataFrame): The training dataset.
        categorical_columns (list of str): List of column names corresponding to categorical features.
        val_df (pd.DataFrame, optional): The validation dataset. Defaults to None.
        feature_ranges (dict, optional): A dictionary specifying the valid ranges for features. 
            Keys are column names, and values are tuples (min, max). Defaults to None.
        k (int, optional): The number of neighbors to use for KNN imputation. Defaults to 15.
        target (str): The name of the target column.

    Returns:
        pd.DataFrame: The processed and normalized training dataset.
        pd.DataFrame, optional: The processed and normalized validation dataset, if provided.
    """
    train_df = train_df.copy()
    val_df = val_df.copy() if val_df is not None else None

    if feature_ranges is not None:
        train_df_imputed = correct_out_of_range_values(train_df, feature_ranges)
        if val_df is not None:
            val_df_imputed = correct_out_of_range_values(val_df, feature_ranges)
    else:
        train_df_imputed = train_df.copy()
        if val_df is not None:
            val_df_imputed = val_df.copy()

    if val_df is not None:
        train_df_imputed, val_df_imputed = knn_impute_with_categorical(train_df_imputed, val_df_imputed)
    else:
        train_df_imputed = knn_impute_with_categorical(train_df_imputed)

    train_df_encoded = one_hot_encode(train_df_imputed, categorical_columns)

    if val_df is not None:
        val_df_encoded = one_hot_encode(val_df_imputed, categorical_columns)

    exclude_columns = [target] + [col for col in train_df_encoded.columns if set(train_df_encoded[col].unique()) <= {0, 1}]

    train_df_normalized, train_mean, train_std = normalize_df(train_df_encoded, train=True, exclude_columns=exclude_columns)

    if val_df is not None:
        val_df_normalized = normalize_df(val_df_encoded, train_mean=train_mean, train_std=train_std, train=False, exclude_columns=exclude_columns)
        return train_df_normalized, val_df_normalized

    return train_df_normalized