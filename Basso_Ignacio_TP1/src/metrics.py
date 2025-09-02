import numpy as np
from IPython.display import display
import pandas as pd


def mse(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) between the true and predicted values.

    Parameters:
    ----------
        y_true (array-like): The ground truth (actual) values.
        y_pred (array-like): The predicted values.

    Returns:
    -------
        float: The mean squared error, calculated as the average of the squared differences 
               between the true and predicted values.
    """
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    """
    Compute the Root Mean Squared Error (RMSE) between the true and predicted values.

    Parameters:
    ----------
    y_true : array-like
        The ground truth (actual) values.
    y_pred : array-like
        The predicted values.

    Returns:
    -------
    float
        The RMSE value, which is always non-negative. A lower value indicates better
        predictive accuracy.
    """
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE) between the true and predicted values.

    Parameters:
    ----------
    y_true (array-like): Array of true values.
    y_pred (array-like): Array of predicted values.

    Returns:
    -------
    float: The mean absolute error.
    """
    return np.mean(np.abs(y_true - y_pred))


def r_dos(y_true, y_pred):
    """
    Calculate the R² (coefficient of determination) score.

    Parameters:
    ----------
    y_true : array-like
        The ground truth (true) values.
    y_pred : array-like
        The predicted values.

    Returns:
    -------
    float
        The R² score. A value of 1 indicates a perfect fit, while a value of 0 
        indicates that the model does not explain any of the variance in the 
        target variable.
    """
    return 1 - mse(y_true, y_pred) / np.var(y_true)


def evaluate_model_performance(X, y, model, set='train', target_modified=0):
    """
    Evaluates the performance of a given model on a dataset and displays the results.
    Parameters:
    -----------
    X : array-like or pandas DataFrame
        The input features used for prediction.
    y : array-like or pandas Series
        The true target values.
    model : object
        The trained model with a `predict` method.
    set : str, optional
        The label for the dataset being evaluated (e.g., 'train', 'test'). Default is 'train'.
    target_modified : int, optional
        Indicates whether the target variable and predictions are in a transformed space (e.g., log-transformed).
        If 0, no transformation is applied. If non-zero, the exponential of the predictions and target values is used.
        Default is 0.
    Returns:
    --------
    None
        Displays a styled DataFrame containing the calculated metrics.
    Metrics:
    --------
    - MSE: Mean Squared Error
    - RMSE: Root Mean Squared Error
    - MAE: Mean Absolute Error
    - R²: Coefficient of Determination (R-squared)
    Notes:
    ------
    - The function assumes that the `mse`, `rmse`, `mae`, and `r_dos` functions are defined elsewhere in the code.
    - The `display` function is used to render the styled DataFrame.
    """
    if target_modified == 0:
        y_pred = model.predict(X)
    else:
        y_pred = np.exp( model.predict(X) )
        y = np.exp(y)
    metrics = {
        "MSE": mse(y, y_pred),
        "RMSE": rmse(y, y_pred),
        "MAE": mae(y, y_pred),
        "R²": r_dos(y, y_pred)
    }
    
    df_metrics = pd.DataFrame(metrics, index=[set.capitalize()])
    display(df_metrics.style.format("{:.2f}"))