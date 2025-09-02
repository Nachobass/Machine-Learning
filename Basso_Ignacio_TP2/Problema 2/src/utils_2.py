import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


def encode_labels(y):
    """
    Encodes categorical labels into integer values.

    This function takes an array of categorical labels and maps each unique label
    to a unique integer. It returns the encoded labels as a NumPy array, along with
    dictionaries for mapping between classes and their corresponding integer values.

    Args:
        y (array-like): An array of categorical labels to be encoded.

    Returns:
        tuple: A tuple containing:
            - y_encoded (numpy.ndarray): An array of integer-encoded labels.
            - class_to_int (dict): A dictionary mapping each class (label) to its corresponding integer.
            - int_to_class (dict): A dictionary mapping each integer back to its corresponding class (label).
    """
    classes = np.unique(y)
    class_to_int = {c: i for i, c in enumerate(classes)}
    int_to_class = {i: c for c, i in class_to_int.items()}
    y_encoded = np.array([class_to_int[label] for label in y])
    return y_encoded, class_to_int, int_to_class

def mutual_information(x, y):
    """
    Calculate the mutual information between two categorical variables.

    Mutual information is a measure of the mutual dependence between two variables.
    It quantifies the amount of information obtained about one variable through the other.

    Parameters:
    ----------
    x : pandas.Series
        A categorical variable.
    y : pandas.Series
        Another categorical variable.

    Returns:
    -------
    float
        The mutual information value between the two variables. A higher value indicates
        greater dependency between the variables.

    Notes:
    -----
    - This function uses a normalized joint probability distribution to compute mutual information.
    - The logarithm used is the natural logarithm (base e).
    """
    joint = pd.crosstab(x, y, normalize=True)
    px = joint.sum(axis=1).values
    py = joint.sum(axis=0).values

    mi = 0.0
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            pxy = joint.iat[i, j]
            if pxy > 0:
                mi += pxy * np.log(pxy / (px[i] * py[j]))
    return mi

def plot_mutual_information_heatmap(mutual_info_df, target, cmap="twilight"):
    """
    Plots a heatmap to visualize the mutual information between features and a target variable.

    Parameters:
    -----------
    mutual_info_df : pandas.DataFrame
        A DataFrame containing mutual information values with columns "Feature" and the corresponding values.
        The "Feature" column should list the feature names, and the other columns should contain the mutual
        information values.
    target : str
        The name of the target variable for which mutual information is calculated.
    cmap : str, optional
        The colormap to use for the heatmap. Default is "twilight".

    Returns:
    --------
    None
        Displays the heatmap plot.

    Notes:
    ------
    - The function adjusts the figure size dynamically based on the number of features.
    - The heatmap includes annotations for mutual information values and a color bar labeled "Mutual Information".
    - The y-axis ticks are removed for a cleaner visualization.
    """
    data_to_plot = mutual_info_df.set_index("Feature").T
    plt.figure(figsize=(max(6, len(data_to_plot.columns)), 2))
    sns.heatmap(data_to_plot, annot=True, fmt=".2f", cmap=cmap,
                cbar_kws={'label': 'Mutual Information'},
                linewidths=1, linecolor='white', annot_kws={"size": 12})
    plt.title(f"Mutual Information with Target: {target}")
    plt.yticks([])
    plt.tight_layout()
    plt.show()

def calculate_mutual_information(df, target, cmap="viridis", bins=10, plot=True):
    """
    Calculate the mutual information between each feature in a DataFrame and the target variable.
    Mutual information measures the dependency between two variables. This function computes
    the mutual information for each feature in the DataFrame with respect to the target variable
    and optionally plots a heatmap of the results.
    Args:
        df (pd.DataFrame): The input DataFrame containing features and the target variable.
        target (str): The name of the target variable column in the DataFrame.
        cmap (str, optional): The colormap to use for the heatmap plot. Defaults to "viridis".
        bins (int, optional): The number of bins to use for discretizing numerical features. Defaults to 10.
        plot (bool, optional): Whether to plot a heatmap of the mutual information scores. Defaults to True.
    Returns:
        pd.DataFrame: A DataFrame containing the features and their corresponding mutual information scores,
                      sorted in descending order of mutual information.
    Notes:
        - Numerical features are discretized into bins using quantile-based discretization before
          calculating mutual information.
        - The function assumes the presence of a `mutual_information` function to compute mutual
          information and a `plot_mutual_information_heatmap` function to generate the heatmap plot.
    """
    X = df.drop(columns=[target])
    y = df[target]
    
    mutual_info_scores = []

    for feature in X.columns:
        col_data = X[feature]

        if np.issubdtype(col_data.dtype, np.number):
            col_data = pd.qcut(col_data, q=bins, duplicates='drop')

        mi = mutual_information(col_data, y)
        mutual_info_scores.append((feature, mi))

    mutual_info_scores = sorted(mutual_info_scores, key=lambda x: x[1], reverse=True)
    mutual_info_df = pd.DataFrame(mutual_info_scores, columns=["Feature", "Mutual Information"])

    print("Mutual Information entre las características y la variable objetivo:")
    display(mutual_info_df)

    if plot:
        plot_mutual_information_heatmap(mutual_info_df, target, cmap=cmap)

    return mutual_info_df