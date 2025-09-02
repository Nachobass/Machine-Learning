import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, X, y, l1=0, l2=0):
        """
        Initializes the model with input data, target values, and regularization parameters.

        Args:
            X (array-like): The input data/features.
            y (array-like): The target values.
            l1 (float, optional): The L1 regularization parameter. Default is 0.
            l2 (float, optional): The L2 regularization parameter. Default is 0.
        """
        self.X = X
        self.y = y
        self.coef = None
        self.l1 = l1
        self.l2 = l2

    def train_pseudo_inverse(self):
        """
        Trains a linear model using the pseudo-inverse method.

        This method computes the coefficients of the linear model by applying
        the pseudo-inverse formula: coef = (X^T * X)^(-1) * X^T * y, where X
        is the design matrix and y is the target vector.

        The computed coefficients are stored in the `self.coef` attribute.

        Returns:
            None
        """
        self.coef = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.y
    
    def train_pseudo_inverse_regularized(self, l2=0):
        """
        Trains a linear model using the regularized pseudo-inverse method.

        This method computes the coefficients of a linear model by applying
        the formula:
            coef = (X^T * X + λ * I)^(-1) * X^T * y
        where λ is the regularization parameter (l2), I is the identity matrix,
        X is the input feature matrix, and y is the target vector.

        Args:
            l2 (float, optional): The L2 regularization parameter (λ). Defaults to 0.

        Attributes:
            coef (numpy.ndarray): The computed coefficients of the linear model.
        """
        I = np.eye(self.X.shape[1])  # Identity matrix
        self.coef = np.linalg.inv(self.X.T @ self.X + l2 * I) @ self.X.T @ self.y
    
    def train_gradient_descent(self, lr=0.0001, epochs=10000):
        """
        Trains a linear regression model using gradient descent.

        Parameters:
            lr (float, optional): Learning rate for gradient descent. Default is 0.0001.
            epochs (int, optional): Number of iterations for training. Default is 10000.

        Attributes:
            X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
            y (numpy.ndarray): Target vector of shape (n_samples,).
            coef (numpy.ndarray): Coefficients of the linear regression model of shape (n_features,).

        Process:
            - Initializes the coefficients to zeros.
            - Iteratively updates the coefficients by computing the gradient of the loss function
              (mean squared error) with respect to the coefficients and adjusting them using the
              learning rate.

        Returns:
            None
        """
        X, y = self.X, self.y
        n = X.shape[0]
        m = X.shape[1]
        self.coef = np.zeros(m)
        for _ in range(epochs):
            y_pred = X @ self.coef
            error = y_pred - y
            gradient = (X.T @ error) / n
            self.coef -= lr * gradient

    def train_gradient_descent_lasso(self, lr=0.0001, epochs=10000,l1=0):
        """
        Trains a Lasso regression model using gradient descent.

        Parameters:
            lr (float, optional): Learning rate for gradient descent. Default is 0.0001.
            epochs (int, optional): Number of iterations for gradient descent. Default is 10000.
            l1 (float, optional): L1 regularization strength (Lasso penalty). Default is 0.

        Attributes:
            X (numpy.ndarray): Feature matrix used for training.
            y (numpy.ndarray): Target vector used for training.
            coef (numpy.ndarray): Coefficients of the trained Lasso regression model.

        Returns:
            None

        Notes:
            - The method initializes the coefficients randomly and updates them iteratively
              using gradient descent with L1 regularization.
            - The training process stops early if the norm of the update step falls below 1e-6.
        """
        X, y = self.X, self.y
        n = X.shape[0]
        m = X.shape[1]
        self.coef = np.random.randn(m) * 0.01

        for _ in range(epochs):
            y_pred = X @ self.coef
            error = y_pred - y
            gradient = (X.T @ error) / n
            self.coef -= lr * (gradient + l1 * np.sign(self.coef))
            if np.linalg.norm(lr * (gradient + l1 * np.sign(self.coef))) < 1e-6:
                break

    def predict(self, X):
        """
        Predicts the output using the linear model.

        Parameters:
        ----------
        X : numpy.ndarray
            A 2D array of shape (n_samples, n_features) representing the input data.

        Returns:
        -------
        numpy.ndarray
            A 1D array of shape (n_samples,) containing the predicted values.
        """
        return X @ self.coef

    def print_coef(model, x_data):
        """
        Prints the coefficients of a given model alongside their corresponding features.

        Args:
            model: The trained model object that contains the coefficients (e.g., a linear regression model).
                   It is expected to have a `coef` attribute.
            x_data: An iterable containing the feature names corresponding to the model's coefficients.

        Returns:
            None. The function prints the coefficients and their associated features in a formatted table.
        """
        print("\nModel coefficients:")
        print("-" * 30)
        for feature, coef in zip(x_data, model.coef):
            print(f"{feature:<20} | {coef:.5f}")
        print("-" * 30)

    def plot_model_performance(self, X_train, y_train, X_validation, y_validation, target):
        """
        Plots the performance of the model by comparing the true and predicted values 
        for both the training and validation datasets.
        Parameters:
        -----------
        X_train : array-like
            Features of the training dataset.
        y_train : array-like
            True target values for the training dataset.
        X_validation : array-like
            Features of the validation dataset.
        y_validation : array-like
            True target values for the validation dataset.
        target : str
            The name of the target variable. If the target is 'log_price', the exponential 
            transformation will be applied to revert the logarithmic scaling.
        Returns:
        --------
        None
            Displays a scatter plot comparing true and predicted prices for both training 
            and validation datasets. The plot includes a reference line (y = x) for visualizing 
            prediction accuracy.
        Notes:
        ------
        - If the target variable is logarithmically transformed (e.g., 'log_price'), the 
          method applies the exponential function to revert the transformation before plotting.
        - The scatter plot uses different colors for training and validation datasets, and 
          includes a legend, grid, and axis labels for clarity.
        """
        y_pred_train = self.predict(X_train)
        y_pred_val = self.predict(X_validation)
        if target == 'log_price':
            y_train = np.exp(y_train)
            y_pred_train = np.exp(y_pred_train)
            y_validation = np.exp(y_validation)
            y_pred_val = np.exp(y_pred_val)
        min_price = min(y_train.min(), y_validation.min())
        max_price = max(y_train.max(), y_validation.max())

        plt.figure(figsize=(10, 6))
        plt.scatter(y_train, y_pred_train, color='violet', label='Train', alpha=0.5)
        plt.scatter(y_validation, y_pred_val, color='blue', label='Validation', alpha=0.5)
        plt.plot([min_price, max_price], [min_price, max_price], 'k--', lw=2)
        plt.xlabel('True price [USD]')
        plt.ylabel('Predicted price [USD]')
        plt.title('Predicted vs. True price', weight='bold')
        plt.legend()
        plt.grid()
        plt.show()