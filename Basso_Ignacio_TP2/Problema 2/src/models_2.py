import numpy as np
from collections import Counter
import math
from itertools import product
from metrics_2 import f1_macro


class LDA:
    def fit(self, X, y):
        """
        Fits the model to the training data.

        This method computes the class-specific means, priors, and shared covariance 
        matrix for the given training data. It assumes that the data follows a 
        Gaussian distribution and that all classes share the same covariance matrix.

        Parameters:
        ----------
        X : numpy.ndarray
            A 2D array of shape (n_samples, n_features) representing the training data.
        y : numpy.ndarray
            A 1D array of shape (n_samples,) containing the class labels for the training data.

        Attributes:
        ----------
        classes_ : numpy.ndarray
            Unique class labels found in the training data.
        means_ : dict
            A dictionary where the keys are class labels and the values are the mean vectors 
            for each class.
        priors_ : dict
            A dictionary where the keys are class labels and the values are the prior 
            probabilities for each class.
        cov_ : numpy.ndarray
            The shared covariance matrix of shape (n_features, n_features).
        cov_inv_ : numpy.ndarray
            The inverse of the shared covariance matrix.
        """
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        self.means_ = {}
        self.priors_ = {}
        self.cov_ = np.zeros((n_features, n_features))

        for c in self.classes_:
            X_c = X[y == c]
            self.means_[c] = np.mean(X_c, axis=0)
            self.priors_[c] = X_c.shape[0] / X.shape[0]
            self.cov_ += np.cov(X_c, rowvar=False) * (X_c.shape[0] - 1)

        self.cov_ /= (X.shape[0] - len(self.classes_))
        self.cov_inv_ = np.linalg.inv(self.cov_)

    def predict(self, X):
        """
        Predict the class labels for the given input data.

        Parameters:
        -----------
        X : numpy.ndarray
            A 2D array of shape (n_samples, n_features) representing the input data.

        Returns:
        --------
        numpy.ndarray
            A 1D array of shape (n_samples,) containing the predicted class labels for each input sample.
        """
        scores = []
        for c in self.classes_:
            mean = self.means_[c]
            prior = np.log(self.priors_[c])
            score = X @ self.cov_inv_ @ mean - 0.5 * mean.T @ self.cov_inv_ @ mean + prior
            scores.append(score)
        return self.classes_[np.argmax(np.vstack(scores), axis=0)]
    
    def predict_proba(self, X):
        """
        Predict class probabilities for the input data.

        Parameters:
        -----------
        X : numpy.ndarray
            A 2D array of shape (n_samples, n_features) representing the input data.
            Each row corresponds to a sample, and each column corresponds to a feature.

        Returns:
        --------
        numpy.ndarray
            A 2D array of shape (n_samples, n_classes) where each element represents
            the predicted probability of the corresponding class for each sample.
            The probabilities are normalized using the softmax function.

        Raises:
        -------
        ValueError
            If the number of features in X does not match the expected number of features
            based on the covariance matrix dimensions.

        Notes:
        ------
        - The method computes a score for each class using the provided means, priors,
          and the inverse covariance matrix.
        - The scores are converted to probabilities using the softmax function to ensure
          they sum to 1 for each sample.
        """
        if X.shape[1] != self.cov_inv_.shape[0]:
            raise ValueError(f"Dimensiones inconsistentes: X tiene {X.shape[1]} características, pero se esperaban {self.cov_inv_.shape[0]}.")
        probs = []
        for c in self.classes_:
            mean = self.means_[c]
            prior = self.priors_[c]
            score = X @ self.cov_inv_ @ mean - 0.5 * mean.T @ self.cov_inv_ @ mean + np.log(prior)
            probs.append(score)
        probs = np.vstack(probs).T
        # Softmax to convert scores to probabilities
        exp_scores = np.exp(probs - np.max(probs, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    



def entropy(y):
    """
    Calculate the entropy of a list of class labels.

    Entropy is a measure of the impurity or randomness in a dataset. It is 
    calculated using the formula:
        H(y) = -Σ(p_i * log2(p_i))
    where p_i is the probability of each unique class in the dataset.

    Args:
        y (list or array-like): A list of class labels.

    Returns:
        float: The entropy value, which is a non-negative number. A value of 
        0 indicates that all elements belong to the same class, while higher 
        values indicate greater impurity or randomness.
    """
    counts = Counter(y)
    probs = [count / len(y) for count in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)

def split(X, y, feature, threshold):
    """
    Splits the dataset into two subsets based on a feature and a threshold.

    Parameters:
    X (numpy.ndarray): The feature matrix of shape (n_samples, n_features).
    y (numpy.ndarray): The target array of shape (n_samples,).
    feature (int): The index of the feature to split on.
    threshold (float): The threshold value to split the feature.

    Returns:
    tuple: A tuple containing four elements:
        - X[left] (numpy.ndarray): Subset of X where the feature values are less than the threshold.
        - y[left] (numpy.ndarray): Corresponding subset of y for X[left].
        - X[right] (numpy.ndarray): Subset of X where the feature values are greater than or equal to the threshold.
        - y[right] (numpy.ndarray): Corresponding subset of y for X[right].
    """
    left = X[:, feature] < threshold
    right = ~left
    return X[left], y[left], X[right], y[right]

def best_split(X, y):
    """
    Finds the best feature and threshold to split the dataset in order to maximize information gain.

    Parameters:
    -----------
    X : numpy.ndarray
        A 2D array of shape (n_samples, n_features) representing the feature matrix.
    y : numpy.ndarray
        A 1D array of shape (n_samples,) representing the target labels.

    Returns:
    --------
    best_feature : int or None
        The index of the feature that provides the best split. Returns None if no valid split is found.
    best_threshold : float or None
        The threshold value for the best split. Returns None if no valid split is found.
    """
    n_features = X.shape[1]
    best_feature, best_threshold, best_gain = None, None, -np.inf
    current_entropy = entropy(y)

    features = np.random.choice(n_features, size=int(np.sqrt(n_features)), replace=False)
    for feature in features:
        thresholds = np.unique(X[:, feature])
        for t in thresholds:
            X_left, y_left, X_right, y_right = split(X, y, feature, t)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            gain = current_entropy - (len(y_left)/len(y))*entropy(y_left) - (len(y_right)/len(y))*entropy(y_right)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = t
    return best_feature, best_threshold


def k_fold_split(X, y, k=5, shuffle=True, seed=42):
    """
    Splits the dataset into k folds for cross-validation.

    Parameters:
    -----------
    X : array-like
        The feature matrix of the dataset.
    y : array-like
        The target vector of the dataset.
    k : int, optional (default=5)
        The number of folds to create.
    shuffle : bool, optional (default=True)
        Whether to shuffle the data before splitting into folds.
    seed : int, optional (default=42)
        The random seed used for shuffling the data.

    Returns:
    --------
    folds : list of tuples
        A list of k tuples, where each tuple contains two arrays:
        - train_idx: The indices for the training set.
        - test_idx: The indices for the test set.

    Notes:
    ------
    - The function ensures that the dataset is split into k folds of approximately equal size.
    - If the dataset size is not perfectly divisible by k, some folds will have one more sample than others.
    """
    np.random.seed(seed)
    indices = np.arange(len(X))
    if shuffle:
        np.random.shuffle(indices)
    fold_sizes = np.full(k, len(X) // k)
    fold_sizes[:len(X) % k] += 1

    folds = []
    start = 0
    for fold_size in fold_sizes:
        stop = start + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_idx, test_idx))
        start = stop
    return folds

def cross_validate_model(ModelClass, X, y, param_grid, k=5, scoring=f1_macro):
    """
    Perform k-fold cross-validation for a given model class and parameter grid.

    Parameters:
    -----------
    ModelClass : class
        The machine learning model class to be instantiated and evaluated.
    X : array-like
        Feature matrix of shape (n_samples, n_features).
    y : array-like
        Target vector of shape (n_samples,).
    param_grid : dict
        Dictionary where keys are parameter names and values are lists of parameter settings to try.
    k : int, optional (default=5)
        Number of folds for k-fold cross-validation.
    scoring : callable, optional (default=f1_macro)
        A scoring function that takes true labels and predicted labels as input and returns a score.

    Returns:
    --------
    dict
        A dictionary containing:
        - "best_params": dict
            The parameter combination that achieved the best score.
        - "best_score": float
            The highest score achieved during cross-validation.
        - "results": list of tuples
            A list of tuples where each tuple contains a parameter combination and its corresponding score.

    Notes:
    ------
    - The function evaluates all combinations of parameters in `param_grid`.
    - The scoring function should be compatible with the output of the model's `predict` method.
    """
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    folds = k_fold_split(X, y, k)
    best_score = -np.inf
    best_params = None
    all_results = []
    all_y_true = []
    all_y_pred = []

    for params in param_combinations:
        for train_idx, test_idx in folds:
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            model = ModelClass(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

        avg_f1_score = scoring(np.array(all_y_true), np.array(all_y_pred))
        all_results.append((params, avg_f1_score))

        if avg_f1_score > best_score:
            best_score = avg_f1_score
            best_params = params

    return {
        "best_params": best_params,
        "best_score": best_score,
        "results": all_results
    }



class TreeNode:
    def __init__(self, depth=0, max_depth=None, min_samples_split=2):
        """
        Initializes a node for a decision tree.

        Parameters:
        depth (int): The depth of the current node in the tree. Default is 0.
        max_depth (int, optional): The maximum depth the tree can grow to. If None, the tree can grow indefinitely. Default is None.
        min_samples_split (int): The minimum number of samples required to split an internal node. Default is 2.

        Attributes:
        depth (int): The depth of the current node.
        max_depth (int or None): The maximum depth of the tree.
        min_samples_split (int): The minimum number of samples required for a split.
        feature (int or None): The index of the feature used for splitting at this node. Default is None.
        threshold (float or None): The threshold value for the feature split. Default is None.
        left (Node or None): The left child node. Default is None.
        right (Node or None): The right child node. Default is None.
        prediction (any or None): The prediction value if the node is a leaf. Default is None.
        """
        self.depth = depth
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.prediction = None

    def fit(self, X, y):
        """
        Fits the decision tree node to the given data.

        This method recursively splits the data based on the best feature and threshold
        until a stopping condition is met. The stopping conditions are:
        - All target labels are the same.
        - The maximum depth of the tree is reached.
        - The number of samples is less than the minimum required to split.

        Parameters:
        ----------
        X : array-like
            The feature matrix of shape (n_samples, n_features).
        y : array-like
            The target labels of shape (n_samples,).

        Returns:
        -------
        None
            The tree node is fitted in place, and the structure of the tree is built
            recursively.
        """
        # Condition to stop splitting
        if len(set(y)) == 1 or self.depth == self.max_depth or len(y) < self.min_samples_split:
            counts = Counter(y)
            total = len(y)
            self.prediction = {cls: counts[cls] / total for cls in counts}
            return
        feat, thresh = best_split(X, y)
        if feat is None:
            self.prediction = Counter(y).most_common(1)[0][0]
            return

        self.feature = feat
        self.threshold = thresh
        X_left, y_left, X_right, y_right = split(X, y, feat, thresh)

        self.left = TreeNode(self.depth + 1, self.max_depth, self.min_samples_split)
        self.left.fit(X_left, y_left)

        self.right = TreeNode(self.depth + 1, self.max_depth, self.min_samples_split)
        self.right.fit(X_right, y_right)

    def predict_one(self, x):
        """
        Predicts the output for a single input instance `x` based on the decision tree logic.

        Args:
            x (dict or array-like): The input instance to predict. It should contain the feature values
                required for the decision tree to make a prediction.

        Returns:
            The predicted value for the input instance `x`. If the prediction is already cached
            in `self.prediction`, it returns that value. Otherwise, it traverses the tree
            based on the feature and threshold values to compute the prediction.
        """
        if self.prediction is not None:
            return self.prediction
        if x[self.feature] < self.threshold:
            return self.left.predict_one(x)
        else:
            return self.right.predict_one(x)


class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, sample_ratio=0.8):
        """
        Initializes the model with the specified hyperparameters.

        Parameters:
        ----------
        n_estimators : int, optional
            The number of trees in the ensemble. Default is 10.
        max_depth : int or None, optional
            The maximum depth of each tree. If None, the trees are expanded until all leaves are pure. Default is None.
        min_samples_split : int, optional
            The minimum number of samples required to split an internal node. Default is 2.
        sample_ratio : float, optional
            The fraction of the dataset to be used for training each tree. Default is 0.8.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.sample_ratio = sample_ratio
        self.trees = []
        self.classes_ = None

    def fit(self, X, y):
        """
        Fits the ensemble model to the training data.

        This method trains multiple decision trees on random subsets of the 
        training data, using bootstrap sampling. Each tree is trained independently 
        and added to the ensemble.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels) corresponding to the input samples.

        Attributes:
        -----------
        classes_ : ndarray of shape (n_classes,)
            The unique class labels found in the target values.

        trees : list of TreeNode
            The list of trained decision trees in the ensemble.

        Notes:
        ------
        - The number of samples used to train each tree is determined by the 
          `sample_ratio` attribute.
        - The trees are trained using the `TreeNode` class, which must implement 
          a `fit` method.
        """
        self.classes_ = np.unique(y)
        n_samples = int(self.sample_ratio * len(X))
        self.trees = []
        for _ in range(self.n_estimators):
            idx = np.random.choice(len(X), n_samples, replace=True)
            tree = TreeNode(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict the class labels for the given input data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data for which predictions are to be made.

        Returns:
        --------
        array-like of shape (n_samples,)
            Predicted class labels for each sample in the input data.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        """
        Predict class probabilities for the input data.
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples for which to predict class probabilities.
        Returns:
        --------
        probs : ndarray of shape (n_samples, n_classes)
            The predicted probabilities for each class. Each row corresponds
            to a sample, and each column corresponds to a class. The values
            represent the average predicted probability for each class across
            all trees in the ensemble.
        Notes:
        ------
        - This method assumes that the `self.trees` attribute contains the
          ensemble of trees and that each tree has a `predict_one` method
          which returns a dictionary of class probabilities for a single sample.
        - The `self.classes_` attribute is expected to contain the list of
          unique class labels, and `self.n_estimators` should represent the
          number of trees in the ensemble.
        """
        class_indices = {c: i for i, c in enumerate(self.classes_)}
        probs = np.zeros((len(X), len(self.classes_)))
        
        for tree in self.trees:
            for i, x in enumerate(X):
                pred_dist = tree.predict_one(x)
                for cls, p in pred_dist.items():
                    probs[i, class_indices[cls]] += p          
        return probs / self.n_estimators