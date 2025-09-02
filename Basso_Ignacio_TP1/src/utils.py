import pandas as pd
import numpy as np
from IPython.display import display
import random


BA_CENTRO = (-34, -58)  # Buenos Aires (Obelisco)
NY_CENTRO = (40, -74)   # Nueva York (Manhattan)


def convert_sqft_to_m2(df):
    """
    Converts area measurements from square feet to square meters in a DataFrame.

    This function modifies the input DataFrame by converting the 'area' column
    from square feet to square meters for rows where the 'area_units' column
    is equal to 'sqft'. It also updates the 'area_units' column to 'm2' for
    those rows.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing at least two columns:
            - 'area': Numeric column representing the area measurements.
            - 'area_units': Column indicating the units of the area ('sqft' or other).

    Returns:
        pandas.DataFrame: The modified DataFrame with area measurements converted
        to square meters where applicable.
    """
    sqft_to_m2 = 0.092903
    df.loc[df['area_units'] == 'sqft', 'area'] *= sqft_to_m2
    df.loc[df['area_units'] == 'sqft', 'area_units'] = 'm2'
    return df


def prepare_data(df, features, target, predict=0):
    """
    Prepares the data for machine learning models by extracting features and target variables,
    and optionally adding an intercept term.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe containing the data.
    features : list of str
        A list of column names to be used as features.
    target : str
        The name of the column to be used as the target variable.
    predict : int, optional, default=0
        If 0, the function returns both the feature matrix (X) and the target vector (y).
        If 1, the function returns only the feature matrix (X).
        If any other value, the function returns None.

    Returns:
    --------
    X : numpy.ndarray
        The feature matrix with an added intercept term (column of ones).
    y : numpy.ndarray, optional
        The target vector. Returned only if `predict` is 0.
    None
        Returned if `predict` is neither 0 nor 1.
    """
    X = df[features].values
    X = np.c_[np.ones((X.shape[0], 1)), X] # Add intercept term
    if predict == 0:
        y = df[target].values
        return X,y
    elif predict == 1:
        return X
    else:
        return


def average_price_per_m2(df):
    """
    Calculate the average price per square meter for houses in the given DataFrame.

    This function filters the DataFrame to include only rows where the 'is_house' column
    is equal to 1, and then computes the average price divided by the average area.

    Parameters:
        df (pandas.DataFrame): A DataFrame containing at least the columns 'is_house',
                               'price', and 'area'.

    Returns:
        float: The average price per square meter for houses. If there are no houses
               in the DataFrame, the result may be NaN.
    """
    return df[df['is_house'] == 1]['price'].mean() / df[df['is_house'] == 1]['area'].mean()


def age_group(age):
    """
    Categorizes an age into a specific group.

    Parameters:
    age (int): The age to be categorized.

    Returns:
    int: The category of the age.
         - Returns 1 if the age is 5 or below.
         - Returns 0 if the age is between 6 and 15 (inclusive).
         - Returns -1 if the age is greater than 15.
    """
    if age <= 5:
        return 1
    elif age <= 15:
        return 0
    else:
        return -1


def apply_age_group(df):
    """
    Applies the `age_group` function to the 'age' column of the given DataFrame.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing an 'age' column.

    Returns:
        pandas.Series: A Series resulting from applying the `age_group` function
        to the 'age' column of the input DataFrame.
    """
    return df['age'].apply(age_group)


def calculate_distance(lat, lon, city_lat, city_lon):
    """
    Calculate the Euclidean distance between two geographical points.

    Parameters:
    lat (float): Latitude of the first point.
    lon (float): Longitude of the first point.
    city_lat (float): Latitude of the second point (e.g., a city).
    city_lon (float): Longitude of the second point (e.g., a city).

    Returns:
    float: The Euclidean distance between the two points.
    """
    return np.sqrt((lat - city_lat)**2 + (lon - city_lon)**2)


def dist_city_center(df):
    """
    Calculate the distance of each row in a DataFrame to a city center.

    This function computes the distance of each row in the input DataFrame to 
    either the Buenos Aires city center (BA_CENTRO) or the New York city center (NY_CENTRO), 
    depending on the latitude value. If the latitude ('lat') is less than 0, the distance 
    is calculated to BA_CENTRO; otherwise, it is calculated to NY_CENTRO.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing at least the following columns:
            - 'lat': Latitude of the location.
            - 'lon': Longitude of the location.

    Returns:
        pd.Series: A pandas Series containing the calculated distances for each row.
    """
    return df.apply(lambda row: 
        calculate_distance(row['lat'], row['lon'], *BA_CENTRO) 
        if row['lat'] < 0 else 
        calculate_distance(row['lat'], row['lon'], *NY_CENTRO), axis=1)


def log_price(df):
    """
    Computes the natural logarithm of the 'price' column in the given DataFrame.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing a 'price' column with numerical values.

    Returns:
    pandas.Series: A Series containing the natural logarithm of the 'price' column.

    Raises:
    KeyError: If the 'price' column is not present in the DataFrame.
    TypeError: If the 'price' column contains non-numerical values.
    """
    return np.log(df['price'])


def log_area(df):
    """
    Computes the natural logarithm of the 'area' column in a given DataFrame.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing an 'area' column with numerical values.

    Returns:
    pandas.Series: A Series containing the natural logarithm of the values in the 'area' column.

    Raises:
    KeyError: If the 'area' column is not present in the DataFrame.
    ValueError: If the 'area' column contains non-positive values, as the logarithm is undefined for such inputs.
    """
    return np.log(df['area'])


def price_density(df):
    """
    Calculate the price density of properties in a DataFrame.

    This function computes the price density by dividing the 'price' column
    by the 'area' column for each row in the given DataFrame.

    Args:
        df (pandas.DataFrame): A DataFrame containing at least two columns:
            'price' (numeric) and 'area' (numeric).

    Returns:
        pandas.Series: A Series containing the price density for each row.

    Raises:
        KeyError: If the DataFrame does not contain the 'price' or 'area' columns.
        ZeroDivisionError: If any value in the 'area' column is zero.
    """
    return df['price'] / df['area']


def rooms_density(df):
    """
    Calculate the density of rooms per unit area for a given DataFrame.

    This function computes the ratio of the number of rooms to the area
    for each row in the provided DataFrame.

    Parameters:
        df (pd.DataFrame): A pandas DataFrame containing at least two columns:
                           'rooms' (number of rooms) and 'area' (total area).

    Returns:
        pd.Series: A pandas Series containing the calculated rooms density
                   for each row in the DataFrame.
    """
    return df['rooms'] / df['area']


def dependence(df, new_feature, function):
    """
    Analyzes the correlation of a newly generated feature with other numerical features in a DataFrame.
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    new_feature : str
        The name of the new feature to be added and analyzed.
    function : callable
        A function that takes the DataFrame as input and generates the values for the new feature.
    Returns:
    --------
    int
        Returns 0 if the new feature has high correlation (absolute value > 0.95) with more than one other variable,
        indicating it is not advisable to use the feature. Returns 1 otherwise.
    Side Effects:
    -------------
    - Prints a warning message if the new feature has high correlation with other variables.
    - Displays a styled DataFrame showing the absolute correlations of the new feature with other variables.
    Notes:
    ------
    - The function creates a copy of the input DataFrame to avoid modifying the original data.
    - Only numerical features are considered for correlation analysis.
    - The function uses a threshold of 0.95 to determine high correlation.
    """
    df_temp = df.copy()
    df_temp[new_feature] = function(df)
    
    corr_matrix = df_temp.select_dtypes(include='number').corr()
    corr_with_others = corr_matrix[new_feature].drop(new_feature)
    
    high_corr_count = corr_with_others[(corr_with_others > 0.95) | (corr_with_others < -0.95)].count()
    
    if high_corr_count > 1:
        print(f'\n🚨 It is not advisable to use the feature: **{new_feature}** due to high correlation with other variables.\n')
        return 0
    else:
        result_table = pd.DataFrame({
            "Variable": corr_with_others.abs().sort_values(ascending=False).index,
            "Correlation": corr_with_others.abs().sort_values(ascending=False).values
        })
        
        print(f'\n**Correlations of {new_feature} with other variables:**\n')
        display(result_table.style.background_gradient(cmap="coolwarm", subset=["Correlation"]))
        return 1


def apply_feature(df, feature, dependence, func):
    """
    Applies a feature transformation to a DataFrame based on a given function.

    Parameters:
        df (pandas.DataFrame): The input DataFrame to which the feature transformation will be applied.
        feature (str): The name of the new feature/column to be added or modified in the DataFrame.
        dependence (int): A flag indicating whether the transformation should be applied (1 to apply, other values to skip).
        func (callable): A function that takes the DataFrame as input and returns the transformed column values.

    Returns:
        pandas.DataFrame: The DataFrame with the new or modified feature if `dependence` is 1; otherwise, the original DataFrame.
    """
    if dependence == 1:
        df[feature] = func(df)
        return df
    else:
        return df
    

def generate_feature_combinations(df, features, num_features=300, min_exp=2, max_exp=35):
    """
    Generates a list of feature-exponent combinations for feature engineering.
    This function creates combinations of features and their corresponding exponents
    within a specified range. If the total number of possible combinations is less
    than the desired number of features (`num_features`), all possible combinations
    are returned. Otherwise, a random sample of the specified size is returned.
    Args:
        df (pd.DataFrame): The input DataFrame (not used in the function, but kept for compatibility).
        features (list): A list of feature names to generate combinations for.
        num_features (int, optional): The number of feature-exponent combinations to generate. Defaults to 300.
        min_exp (int, optional): The minimum exponent value for the combinations. Defaults to 2.
        max_exp (int, optional): The maximum exponent value for the combinations. Defaults to 35.
    Returns:
        list: A list of tuples, where each tuple contains a feature name and an exponent value.
    Notes:
        - If the total number of possible combinations is less than `num_features`, a warning
          is printed, and all possible combinations are returned.
        - The function uses random sampling to select combinations when the number of possible
          combinations exceeds `num_features`.
    """
    all_combinations = [(feat, exp) for feat in features for exp in range(min_exp, max_exp + 1)]
    
    num_possible = len(all_combinations)
    if num_possible < num_features:
        print(f"⚠️ Solo hay {num_possible} combinaciones posibles, usando todas.")
        return all_combinations
    else:
        return random.sample(all_combinations, num_features)


def generate_features(df, selected_combinations):
    """
    Generate new features for a DataFrame by applying exponentiation to selected columns.
    Args:
        df (pd.DataFrame): The input DataFrame containing the original features.
        selected_combinations (list of tuples): A list of tuples where each tuple contains:
            - feature (str): The name of the column in the DataFrame to be transformed.
            - exp (int or float): The exponent to which the column values will be raised.
    Returns:
        tuple:
            - pd.DataFrame: A new DataFrame with the original features and the newly generated features.
            - list of str: A list of the names of the newly generated features.
    """
    new_features = {}
    for feature, exp in selected_combinations:
        col_name = f"{feature}^{exp}"
        new_features[col_name] = df[feature] ** exp
    
    new_features_df = pd.DataFrame(new_features, index=df.index)
    return pd.concat([df, new_features_df], axis=1), list(new_features.keys())