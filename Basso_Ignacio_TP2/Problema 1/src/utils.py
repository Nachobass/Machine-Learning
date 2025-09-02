import numpy as np
import pandas as pd
from models import *
from metrics import f1_score
from preprocessing import process_data, prepare_data



def count_outliers(column):
    """
    Counts the number of outliers in a given pandas Series based on the Interquartile Range (IQR) method.

    An outlier is defined as a value that falls below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR,
    where Q1 is the 25th percentile, Q3 is the 75th percentile, and IQR is the interquartile range (Q3 - Q1).

    Parameters:
        column (pandas.Series): The input column for which outliers are to be counted.

    Returns:
        int: The number of outliers in the column.
    """
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return ((column < lower_bound) | (column > upper_bound)).sum()


def get_feature_ranges(df, numeric_vars, q1, q3):
    """
    Calculate the range of valid values for numeric features in a DataFrame 
    based on the interquartile range (IQR) method.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        numeric_vars (list): A list of column names corresponding to numeric features.
        q1 (float): The lower quantile (e.g., 0.25 for the 25th percentile).
        q3 (float): The upper quantile (e.g., 0.75 for the 75th percentile).

    Returns:
        dict: A dictionary where keys are column names from `numeric_vars` and 
              values are tuples representing the minimum and maximum valid values 
              for each feature, rounded to two decimal places.

    Notes:
        - Negative values in the numeric columns are replaced with NaN before 
          calculating the IQR.
        - The valid range for each feature is determined as:
          [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR], where IQR = Q3 - Q1.
        - Only values within the valid range are considered when calculating 
          the minimum and maximum values for each feature.
    """
    feature_ranges = {}
    df_copy = df.copy()

    for col in numeric_vars:
        if col in df_copy.columns:
            # Replace negative values with NaN
            df_copy[col] = df_copy[col].where(df_copy[col] >= 0, np.nan)

            Q1 = df_copy[col].quantile(q1)
            Q3 = df_copy[col].quantile(q3)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Filter data within the bounds
            valid_values = df_copy[col][(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]

            min_val = valid_values.min()
            max_val = valid_values.max()
            feature_ranges[col] = (round(min_val, 2), round(max_val, 2))

    return feature_ranges


def find_best_lambda(X_train, y_train, X_val, y_val, lambdas, refinement_steps=10, class_weights=None):
    """
    Finds the best regularization parameter (lambda) for a logistic regression model 
    using a two-step process: an initial coarse search followed by a refined search.

    Parameters:
    -----------
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        Training target vector.
    X_val : array-like
        Validation feature matrix.
    y_val : array-like
        Validation target vector.
    lambdas : list or array-like
        List of lambda values to evaluate during the coarse search.
    refinement_steps : int, optional (default=10)
        Number of lambda values to evaluate during the refined search.
    class_weights : dict or None, optional (default=None)
        Class weights to handle imbalanced datasets. If None, no class weights are used.

    Returns:
    --------
    best_model_refined : LogisticRegressionL2
        The logistic regression model trained with the best lambda.
    best_lambda_refined : float
        The best lambda value found during the search.
    best_f1_refined : float
        The F1-score corresponding to the best lambda.

    Notes:
    ------
    - The function uses F1-score as the evaluation metric to select the best lambda.
    - The refined search is performed in the range between the best and second-best 
      lambdas found during the coarse search.
    - The LogisticRegressionL2 class is assumed to be defined elsewhere and supports 
      L2 regularization with a lambda parameter.
    """
    # First search: find the best and second best lambda
    best_f1 = -1
    second_best_f1 = -1
    best_lambda = None
    second_best_lambda = None

    for l in lambdas:
        if class_weights is not None:
            model = LogisticRegressionL2(lambda_=l, class_weights=class_weights)
        else:
            model = LogisticRegressionL2(lambda_=l)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            second_best_f1 = best_f1
            second_best_lambda = best_lambda
            best_f1 = f1
            best_lambda = l
        elif f1 > second_best_f1:
            second_best_f1 = f1
            second_best_lambda = l

    print(f"Primer barrido: Mejor lambda: {best_lambda} con F1-score: {best_f1:.4f}")
    print(f"Segundo mejor lambda: {second_best_lambda} con F1-score: {second_best_f1:.4f}")

    # Second search: refine the best lambda
    refined_lambdas = np.linspace(min(best_lambda, second_best_lambda), max(best_lambda, second_best_lambda), refinement_steps)
    best_f1_refined = -1
    best_lambda_refined = None
    best_model_refined = None

    for l in refined_lambdas:
        model = LogisticRegressionL2(lambda_=l)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1_refined:
            best_f1_refined = f1
            best_lambda_refined = l
            best_model_refined = model

    print(f"Barrido fino: Mejor lambda: {best_lambda_refined} con F1-score: {best_f1_refined:.4f}")
    return best_model_refined, best_lambda_refined, best_f1_refined


def cross_val_f1_score_df(df,lambdas,features,target,categorical_columns,numeric_columns,q1,q3,k=5,seed=42,resampling_fn=None,class_weights=None):
    """
    Perform k-fold cross-validation to evaluate the F1-score of a logistic regression model 
    with L2 regularization for different values of the regularization parameter (lambda). 
    Returns the best model, the best lambda, and the corresponding F1-score.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataset containing features and target variable.
    lambdas : list or array-like
        A list of lambda values (regularization strengths) to evaluate.
    features : list
        List of feature column names to be used for training the model.
    target : str
        The name of the target column in the dataset.
    categorical_columns : list
        List of categorical column names in the dataset.
    numeric_columns : list
        List of numeric column names in the dataset.
    q1 : float
        The first quantile value used for feature scaling.
    q3 : float
        The third quantile value used for feature scaling.
    k : int, optional (default=5)
        The number of folds for cross-validation.
    seed : int, optional (default=42)
        Random seed for reproducibility.
    resampling_fn : callable, optional (default=None)
        A function to perform resampling on the training data (e.g., oversampling or undersampling).
        Should accept (X_train, y_train) as input and return resampled (X_train, y_train).
    class_weights : dict or None, optional (default=None)
        Class weights to handle class imbalance. If None, no class weights are applied.

    Returns:
    --------
    best_model : LogisticRegressionL2
        The logistic regression model with the best F1-score.
    best_lambda : float
        The lambda value corresponding to the best F1-score.
    best_f1 : float
        The highest F1-score achieved during cross-validation.

    Notes:
    ------
    - The function assumes the existence of helper functions `get_feature_ranges`, `process_data`, 
        and `prepare_data` for preprocessing the data.
    - The `LogisticRegressionL2` class is assumed to be a custom implementation of logistic regression 
        with L2 regularization.
    - The F1-score is computed using the `f1_score` function from `sklearn.metrics`.
    """
    np.random.seed(seed)
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    fold_size = len(df) // k

    best_lambda = None
    best_f1 = -1
    best_model = None

    for l in lambdas:
        y_true_all = []
        y_pred_all = []

        for fold in range(k):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < k - 1 else len(df)
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate((indices[:val_start], indices[val_end:]))

            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()

            feature_ranges = get_feature_ranges(train_df, numeric_columns, q1=q1, q3=q3)

            train_df_proc, val_df_proc = process_data(train_df, categorical_columns, val_df=val_df, feature_ranges=feature_ranges)

            X_train, y_train = prepare_data(train_df_proc, features, target)
            X_val, y_val = prepare_data(val_df_proc, features, target)

            # Use resampling function if provided
            if resampling_fn is not None:
                X_train, y_train = resampling_fn(X_train, y_train)

            if class_weights is not None:
                model = LogisticRegressionL2(lambda_=l, class_weights=class_weights)
            else:
                model = LogisticRegressionL2(lambda_=l)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            y_true_all.extend(y_val)
            y_pred_all.extend(y_pred)

        f1 = f1_score(np.array(y_true_all), np.array(y_pred_all))
        print(f"Lambda {l:.5f} - F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_lambda = l
            best_model = model

    print(f"Mejor lambda: {best_lambda} con F1-score: {best_f1:.4f}")
    return best_model, best_lambda, best_f1


def undersampling(X, y):
    """
    Perform undersampling to balance the dataset by reducing the majority class.

    This function takes a dataset (X) and its corresponding labels (y), and reduces
    the number of samples in the majority class to match the number of samples in
    the minority class. It ensures that the resulting dataset has a balanced class
    distribution.

    Parameters:
    -----------
    X : pandas.DataFrame or array-like
        The feature matrix containing the input data. If not a pandas DataFrame,
        it will be converted to one.
    y : pandas.Series or array-like
        The target labels corresponding to the input data. If not a pandas Series,
        it will be converted to one.

    Returns:
    --------
    X_resampled : pandas.DataFrame
        The resampled feature matrix with balanced class distribution.
    y_resampled : pandas.Series
        The resampled target labels with balanced class distribution.

    Notes:
    ------
    - The function assumes that `y` contains at least two classes.
    - The sampling is performed randomly without replacement for the majority class.
    - The minority class is not altered during the resampling process.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X.index)

    class_counts = pd.Series(y).value_counts()
    min_class = class_counts.idxmin()

    min_class_indices = y[y == min_class].index
    maj_class_indices = y[y != min_class].index

    # Choose randomly from the majority class 
    maj_class_sampled_indices = np.random.choice(maj_class_indices, size=len(min_class_indices), replace=False)

    # Mix the indices of both classes
    sampled_indices = np.concatenate([min_class_indices, maj_class_sampled_indices])

    return X.loc[sampled_indices], y.loc[sampled_indices]


def oversampling_duplicate(X, y):    
    """
    Perform oversampling by duplicating samples from the minority class to balance the dataset.

    Parameters:
    -----------
    X : pandas.DataFrame or array-like
        The feature matrix. If not a pandas DataFrame, it will be converted to one.
    y : pandas.Series or array-like
        The target labels. If not a pandas Series, it will be converted to one.

    Returns:
    --------
    X_resampled : pandas.DataFrame
        The resampled feature matrix with balanced classes.
    y_resampled : pandas.Series
        The resampled target labels with balanced classes.

    Notes:
    ------
    - The function identifies the majority and minority classes based on the class distribution in `y`.
    - Samples from the minority class are randomly duplicated with replacement to match the number of samples in the majority class.
    - The resulting dataset contains an equal number of samples for both classes.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X.index)

    class_counts = y.value_counts()
    maj_class = class_counts.idxmax()

    min_class_indices = y[y != maj_class].index
    maj_class_indices = y[y == maj_class].index

    # Choose randomly from the minority class 
    min_class_sampled_indices = np.random.choice(min_class_indices, size=len(maj_class_indices), replace=True)

    # Mix the indices of both classes
    sampled_indices = np.concatenate([maj_class_indices, min_class_sampled_indices])

    return X.loc[sampled_indices], y.loc[sampled_indices]


def oversampling_smote(X, y, k_neighbors=5):
    """
    Perform Synthetic Minority Oversampling Technique (SMOTE) to balance the dataset.

    This function generates synthetic samples for the minority class by interpolating
    between existing minority class samples and their nearest neighbors from the majority class.

    Parameters:
    -----------
    X : pandas.DataFrame or array-like
        Feature matrix containing the input data. If not a pandas DataFrame, it will be converted to one.
    y : pandas.Series or array-like
        Target vector containing class labels. If not a pandas Series, it will be converted to one.
    k_neighbors : int, optional (default=5)
        Number of nearest neighbors to consider when generating synthetic samples.

    Returns:
    --------
    X_balanced : pandas.DataFrame
        Feature matrix after oversampling, containing both original and synthetic samples.
    y_balanced : pandas.Series
        Target vector after oversampling, containing labels for both original and synthetic samples.

    Notes:
    ------
    - This implementation assumes that the dataset contains only two classes, with one being the majority class
      and the other being the minority class.
    - The function uses Euclidean distance to find the nearest neighbors.
    - The synthetic samples are generated by linear interpolation between a minority class sample and its
      nearest neighbors from the majority class.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X.index)

    class_counts = y.value_counts()
    maj_class = class_counts.idxmax()

    min_class_indices = y[y != maj_class].index
    maj_class_indices = y[y == maj_class].index

    new_samples = []

    for idx in min_class_indices:
        # Get the benchmark (minority class sample)
        reference_point = X.loc[idx]

        # Calculate distances to all samples in the majority class
        distances = np.linalg.norm(X.loc[maj_class_indices] - reference_point, axis=1)

        # Find the k nearest neighbors
        nearest_indices = np.argsort(distances)[:k_neighbors]

        # Generate new samples by interpolating between the reference point and its nearest neighbors
        for neighbor_idx in nearest_indices:
            neighbor_point = X.loc[maj_class_indices].iloc[neighbor_idx]
            new_sample = reference_point + np.random.rand() * (neighbor_point - reference_point)
            new_samples.append(new_sample)

    new_samples_df = pd.DataFrame(new_samples, columns=X.columns)
    X_balanced = pd.concat([X, new_samples_df], ignore_index=True)
    y_balanced = pd.concat([y, pd.Series([y[min_class_indices[0]]] * len(new_samples), index=new_samples_df.index)], ignore_index=True)

    return X_balanced, y_balanced