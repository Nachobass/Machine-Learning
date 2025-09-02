import numpy as np


class LogisticRegressionL2:
    def __init__(self, lr=0.01, lambda_=0.0001, num_iter=10000, multi_class='ovr', tol=1e-4, class_weights=None):
        """
        Initializes the model with the specified hyperparameters.

        Parameters:
        ----------
        lr : float, optional
            Learning rate for the optimization algorithm (default is 0.01).
        lambda_ : float, optional
            Regularization strength to prevent overfitting (default is 0.0001).
        num_iter : int, optional
            Number of iterations for the optimization algorithm (default is 10000).
        multi_class : {'ovr', 'multinomial'}, optional
            Strategy for handling multi-class classification. 'ovr' stands for one-vs-rest,
            and 'multinomial' is for softmax regression (default is 'ovr').
        tol : float, optional
            Tolerance for stopping criteria. If the change in loss is smaller than this value,
            the optimization stops (default is 1e-4).
        class_weights : dict or None, optional
            Weights associated with classes to handle class imbalance. If None, all classes
            are assumed to have equal weight (default is None).

        Attributes:
        ----------
        weights : ndarray or None
            Coefficients of the model, initialized during training.
        bias : float or None
            Bias term of the model, initialized during training.
        """
        self.lr = lr
        self.lambda_ = lambda_
        self.num_iter = num_iter
        self.multi_class = multi_class
        self.tol = tol
        self.class_weights = class_weights
        self.weights = None
        self.bias = None


    def sigmoid(self, z):
        """
        Compute the sigmoid function for the input array.

        The sigmoid function is defined as:
            sigmoid(z) = 1 / (1 + exp(-z))
        This implementation includes clipping of the input `z` to avoid numerical
        overflows when computing the exponential.

        Parameters:
        -----------
        z : array-like or scalar
            The input value(s) for which to compute the sigmoid function. Can be a
            scalar, vector, or matrix.

        Returns:
        --------
        numpy.ndarray or scalar
            The sigmoid of the input `z`, with the same shape as the input.
        """
        z = np.clip(z, -500, 500)  # Avoid overflows
        return 1 / (1 + np.exp(-z))
    
    def softmax(self, z):
        """
        Compute the softmax of a given input array.

        The softmax function is often used in machine learning for converting
        logits or raw model outputs into probabilities. It ensures that the
        output values are in the range [0, 1] and sum to 1 across the specified axis.

        Parameters:
            z (numpy.ndarray): Input array of shape (n_samples, n_features), 
                               where n_samples is the number of samples and 
                               n_features is the number of features or classes.

        Returns:
            numpy.ndarray: An array of the same shape as `z`, where each row 
                           represents the softmax probabilities for the corresponding 
                           input row.

        Notes:
            - The computation includes a numerical stability trick by subtracting 
              the maximum value in each row from the input array `z` before 
              exponentiation. This prevents potential overflow issues when 
              exponentiating large numbers.
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Avoid numeric problems
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        """
        Fit the model to the training data.
        Parameters:
        -----------
        X : numpy.ndarray
            The input features of shape (m, n), where m is the number of samples and n is the number of features.
        y : numpy.ndarray
            The target labels of shape (m,).
        Returns:
        --------
        None
        Notes:
        ------
        - This method supports two modes for multi-class classification:
            - 'ovr' (One-vs-Rest): Trains a binary classifier for each class.
            - 'softmax': Uses the softmax function for multi-class classification.
        - The method applies gradient descent to optimize weights and biases.
        - Supports optional class weighting for imbalanced datasets.
        - Includes L2 regularization to prevent overfitting.
        - The learning rate decays slightly after each epoch.
        """
        m, n = X.shape
        self.classes_ = np.unique(y)
        
        if self.multi_class == 'ovr':
            self.weights = np.random.randn(len(self.classes_), n) * 0.01
            self.bias = np.zeros(len(self.classes_))

            for i, c in enumerate(self.classes_):
                y_binary = (y == c).astype(int)
                w, b = self.weights[i], self.bias[i]
                
                for epoch in range(self.num_iter):
                    # Prediction
                    z = np.dot(X, w) + b
                    y_pred = self.sigmoid(z)

                    # Cost re-weighting
                    if self.class_weights:
                        w0 = self.class_weights.get(0, 1.0)
                        w1 = self.class_weights.get(1, 1.0)
                        sample_weights = np.where(y_binary == 1, w1, w0)
                    else:
                        sample_weights = np.ones_like(y_binary)

                    # Reweighted loss
                    loss = (-sample_weights * (y_binary * np.log(y_pred + 1e-8) + 
                                               (1 - y_binary) * np.log(1 - y_pred + 1e-8))).mean()
                    loss += (self.lambda_ / (2 * m)) * np.sum(w ** 2)

                    # Reweighted gradients
                    error = y_pred - y_binary
                    dw = (X.T @ (sample_weights * error)) / m + (self.lambda_ * w / m)
                    db = np.sum(sample_weights * error) / m

                    w -= self.lr * dw
                    b -= self.lr * db
                    self.lr *= 0.9995

                self.weights[i] = w
                self.bias[i] = b

        elif self.multi_class == 'softmax':
            self.weights = np.random.randn(len(self.classes_), n) * 0.01
            self.bias = np.zeros(len(self.classes_))

            y_one_hot = np.zeros((m, len(self.classes_)))
            for i, c in enumerate(self.classes_):
                y_one_hot[:, i] = (y == c).astype(int)

            for epoch in range(self.num_iter):
                z = np.dot(X, self.weights.T) + self.bias
                y_pred = self.softmax(z)

                loss = -np.mean(np.sum(y_one_hot * np.log(y_pred + 1e-8), axis=1))
                loss += (self.lambda_ / (2 * m)) * np.sum(self.weights ** 2)

                dw = np.dot((y_pred - y_one_hot).T, X) / m + (self.lambda_ * self.weights / m)
                db = np.sum(y_pred - y_one_hot, axis=0) / m

                self.weights -= self.lr * dw
                self.bias -= self.lr * db
                self.lr *= 0.9995

    def predict_proba(self, X):
        """
        Predict the probability estimates for the input data.

        Parameters:
        -----------
        X : numpy.ndarray
            The input data matrix of shape (n_samples, n_features).

        Returns:
        --------
        numpy.ndarray
            The predicted probabilities for each class. If `multi_class` is set to 'ovr',
            the probabilities are computed using the sigmoid function. Otherwise, the
            probabilities are computed using the softmax function.
        """
        scores = np.dot(X, self.weights.T) + self.bias
        if self.multi_class == 'ovr':
            return self.sigmoid(scores)
        return self.softmax(scores)

    def predict(self, X):
        """
        Predict the class labels for the given input data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data for which predictions are to be made.

        Returns:
        --------
        array of shape (n_samples,)
            Predicted class labels for each sample in the input data.
            The class label corresponds to the index of the highest probability
            predicted by the model.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)  # Returns the class with the highest probability
    
    def print__coef(model, x_data, class_names=None):
        """
        Prints the coefficients of a model for each class in a readable format.

        Args:
            model: An object representing the trained model. It must have a `weights` attribute,
                   which is a 2D array where each row corresponds to the coefficients for a class.
            x_data: A list of feature names corresponding to the features used in the model.
            class_names (list, optional): A list of class names. If not provided, default class
                                          names will be generated as "Class 0", "Class 1", etc.

        Prints:
            A formatted table of coefficients for each class, showing the feature names and their
            corresponding coefficient values.
        """
        n_classes = model.weights.shape[0]
        if class_names is None:
            class_names = [f"Class {i}" for i in range(n_classes)]

        print("\nModel coefficients per class:")
        print("=" * 50)
        for class_idx, class_name in enumerate(class_names):
            print(f"\n{class_name} coefficients:")
            print("-" * 50)
            for feature, coef in zip(x_data, model.weights[class_idx]):
                print(f"{feature:<20} | {coef:.5f}")
            print("-" * 50)