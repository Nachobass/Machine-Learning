import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from IPython.display import display
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
sys.path.append('src')
from pytorch_models2 import *



def relu(Z):
    """
    Applies the Rectified Linear Unit (ReLU) activation function.

    The ReLU function sets all negative values in the input array to zero 
    and leaves all non-negative values unchanged.

    Parameters:
    Z (numpy.ndarray): Input array or matrix to which the ReLU function will be applied.

    Returns:
    numpy.ndarray: An array with the same shape as `Z`, where all negative values 
                   have been replaced with zero.
    """
    return np.maximum(0, Z)


def relu_derivative(Z):
    """
    Computes the derivative of the ReLU (Rectified Linear Unit) activation function.

    The ReLU derivative is 1 for positive input values and 0 for non-positive input values.

    Parameters:
    Z (numpy.ndarray): Input array for which the derivative of ReLU is to be computed.

    Returns:
    numpy.ndarray: An array of the same shape as Z, where each element is the derivative
                   of ReLU applied to the corresponding element in Z.
    """
    return (Z > 0).astype(float)


def softmax(Z):
    """
    Computes the softmax function for each row of the input array.

    The softmax function is often used in machine learning for converting
    logits (raw model outputs) into probabilities. It normalizes the input
    values such that they sum to 1 along the specified axis.

    Parameters:
    -----------
    Z : numpy.ndarray
        A 2D array of shape (n_samples, n_features) where each row represents
        a set of scores or logits.

    Returns:
    --------
    numpy.ndarray
        A 2D array of the same shape as `Z`, where each row contains the
        softmax probabilities corresponding to the input scores.

    Notes:
    ------
    - The function applies a numerical stability trick by subtracting the
      maximum value in each row from all elements in that row before
      exponentiating. This prevents potential overflow issues with large
      input values.
    - The output probabilities for each row sum to 1.
    """
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)


def initialize_parameters(layer_dims):
    """
    Initializes the parameters for a neural network with given layer dimensions.

    Args:
        layer_dims (list): A list containing the dimensions of each layer in the network.
                           For example, [input_size, hidden_layer_1_size, ..., output_size].

    Returns:
        dict: A dictionary containing the initialized parameters:
              - 'Wl': Weights matrix of shape (layer_dims[l-1], layer_dims[l]) for layer l,
                      initialized using He initialization.
              - 'bl': Bias vector of shape (1, layer_dims[l]) for layer l, initialized to zeros.
    """
    parameters = {}
    for l in range(1, len(layer_dims)):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l-1], layer_dims[l]) * np.sqrt(2. / layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((1, layer_dims[l]))
    return parameters


def initialize_adam(parameters):
    """
    Initializes the Adam optimization variables (v and s) for the given parameters.
    Adam optimization requires two sets of variables:
    - v: Exponentially weighted average of the gradients (momentum term).
    - s: Exponentially weighted average of the squared gradients (RMSProp term).
    Args:
        parameters (dict): Dictionary containing the parameters of the model. 
                           Keys are "W1", "b1", ..., "WL", "bL" where L is the number of layers.
    Returns:
        tuple: A tuple (v, s) where:
            - v (dict): Dictionary containing the initialized momentum terms for gradients.
                        Keys are "dW1", "db1", ..., "dWL", "dbL".
            - s (dict): Dictionary containing the initialized RMSProp terms for gradients.
                        Keys are "dW1", "db1", ..., "dWL", "dbL".
    """
    L = len(parameters) // 2
    v = {}
    s = {}
    
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
        s["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        s["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])

    return v, s


def forward_propagation(X, parameters, keep_prob=1.0):
    """
    Implements the forward propagation for a neural network with optional dropout regularization.

    Args:
        X (numpy.ndarray): Input data of shape (number of examples, number of features).
        parameters (dict): Dictionary containing the weights and biases of the neural network.
            - 'W1', 'W2', ..., 'WL': Weight matrices for each layer.
            - 'b1', 'b2', ..., 'bL': Bias vectors for each layer.
        keep_prob (float, optional): Probability of keeping a neuron active during dropout. 
            Defaults to 1.0 (no dropout).

    Returns:
        tuple: 
            - AL (numpy.ndarray): The output of the last layer (predictions).
            - cache (dict): Dictionary containing intermediate values (activations and linear combinations)
              for each layer, useful for backpropagation.
            - dropout_masks (dict, optional): Dictionary containing dropout masks for each layer 
              (only returned if keep_prob < 1.0).
    """
    cache = {'A0': X}
    L = len(parameters) // 2
    dropout_masks = {} if keep_prob < 1.0 else None

    for l in range(1, L):
        Z = cache['A' + str(l-1)] @ parameters['W' + str(l)] + parameters['b' + str(l)]
        A = relu(Z)

        if keep_prob < 1.0:
            D = (np.random.rand(*A.shape) < keep_prob).astype(float)
            A *= D
            A /= keep_prob
            dropout_masks['D' + str(l)] = D

        cache['Z' + str(l)] = Z
        cache['A' + str(l)] = A

    # output layer
    ZL = cache['A' + str(L-1)] @ parameters['W' + str(L)] + parameters['b' + str(L)]
    AL = softmax(ZL)
    cache['Z' + str(L)] = ZL
    cache['A' + str(L)] = AL

    if keep_prob < 1.0:
        return AL, cache, dropout_masks
    else:
        return AL, cache


def compute_cost(AL, Y, parameters=None, lambd=0):
    """
    Computes the cost function for a neural network, including an optional L2 regularization term.

    Parameters:
    AL (numpy.ndarray): The predicted probabilities (output of the forward propagation), 
                        shape (m, n_classes), where m is the number of examples.
    Y (numpy.ndarray): The true labels, shape (m, n_classes), where m is the number of examples.
    parameters (dict, optional): Dictionary containing the weights of the neural network. 
                                  Used for L2 regularization. Default is None.
    lambd (float, optional): Regularization hyperparameter. Default is 0 (no regularization).

    Returns:
    float: The computed cost, which includes the cross-entropy loss and, if applicable, 
           the L2 regularization term.
    """
    m = Y.shape[0]
    cross_entropy = -np.sum(Y * np.log(AL + 1e-8)) / m

    if parameters is not None and lambd > 0:
        L2_regularization = 0
        L = len(parameters) // 2
        for l in range(1, L + 1):
            W = parameters['W' + str(l)]
            L2_regularization += np.sum(W ** 2)
        L2_term = (lambd / (2 * m)) * L2_regularization
        return cross_entropy + L2_term

    return cross_entropy


def backward_propagation(parameters, cache, X, Y, lambd=0, dropout_masks=None, keep_prob=1.0):
    """
    Implements the backward propagation for a neural network with optional L2 regularization 
    and dropout.
    Args:
        parameters (dict): Dictionary containing the weights and biases of the neural network.
            - 'W1', 'W2', ..., 'WL': Weight matrices for each layer.
            - 'b1', 'b2', ..., 'bL': Bias vectors for each layer.
        cache (dict): Dictionary containing intermediate values from forward propagation.
            - 'A0', 'A1', ..., 'AL': Activations for each layer.
        X (numpy.ndarray): Input data of shape (m, n_x), where m is the number of examples 
            and n_x is the number of features.
        Y (numpy.ndarray): True labels of shape (m, n_y), where m is the number of examples 
            and n_y is the number of output classes.
        lambd (float, optional): L2 regularization hyperparameter. Defaults to 0 (no regularization).
        dropout_masks (dict, optional): Dictionary containing dropout masks for each layer 
            during forward propagation. Defaults to None (no dropout).
        keep_prob (float, optional): Probability of keeping a neuron active during dropout. 
            Defaults to 1.0 (no dropout).
    Returns:
        dict: Dictionary containing gradients for weights and biases of each layer.
            - 'dW1', 'dW2', ..., 'dWL': Gradients of the weight matrices.
            - 'db1', 'db2', ..., 'dbL': Gradients of the bias vectors.
    """
    grads = {}
    L = len(parameters) // 2
    m = X.shape[0]
    
    # output layer: softmax + cross-entropy
    A_L = cache['A' + str(L)]
    dZ = A_L - Y
    grads['dW' + str(L)] = (cache['A' + str(L-1)].T @ dZ) / m
    grads['db' + str(L)] = np.sum(dZ, axis=0, keepdims=True) / m
    if lambd > 0:
        grads['dW' + str(L)] += (lambd / m) * parameters['W' + str(L)]
    
    # hidden layers
    for l in reversed(range(1, L)):
        A_prev = cache['A' + str(l-1)]
        Z = cache['A' + str(l)]
        dA = dZ @ parameters['W' + str(l+1)].T
        dZ = dA * relu_derivative(Z)

        # dropout application if keep_prob < 1
        if keep_prob < 1.0 and dropout_masks is not None:
            D = dropout_masks['D' + str(l)]
            dZ *= D  # apply same mask as in forward
            dZ /= keep_prob

        grads['dW' + str(l)] = (A_prev.T @ dZ) / m
        grads['db' + str(l)] = np.sum(dZ, axis=0, keepdims=True) / m
        if lambd > 0:
            grads['dW' + str(l)] += (lambd / m) * parameters['W' + str(l)]

    return grads


def update_parameters(parameters, grads, learning_rate,
                      v=None, s=None, t=None,
                      beta1=0, beta2=0, epsilon=0):
    """
    Update parameters using gradient descent or Adam optimization.

    Args:
        parameters (dict): Dictionary containing the parameters "W1", "b1", ..., "WL", "bL".
        grads (dict): Dictionary containing the gradients "dW1", "db1", ..., "dWL", "dbL".
        learning_rate (float): Learning rate for the update step.
        v (dict, optional): Dictionary containing the exponentially weighted average of past gradients 
                            (used for momentum in Adam). Defaults to None.
        s (dict, optional): Dictionary containing the exponentially weighted average of past squared gradients 
                            (used for RMSProp in Adam). Defaults to None.
        t (int, optional): Current iteration number (used for bias correction in Adam). Defaults to None.
        beta1 (float, optional): Exponential decay hyperparameter for the first moment estimates in Adam. Defaults to 0.
        beta2 (float, optional): Exponential decay hyperparameter for the second moment estimates in Adam. Defaults to 0.
        epsilon (float, optional): Small value to prevent division by zero in Adam. Defaults to 0.

    Returns:
        tuple: 
            - parameters (dict): Updated parameters.
            - v (dict, optional): Updated first moment estimates (if Adam is used).
            - s (dict, optional): Updated second moment estimates (if Adam is used).
    """

    L = len(parameters) // 2

    if v is not None and s is not None and t is not None:
        # adam
        v_corrected = {}
        s_corrected = {}

        for l in range(1, L + 1):
            # momentum
            v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
            v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]

            # RMSProp
            s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * (grads["dW" + str(l)] ** 2)
            s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * (grads["db" + str(l)] ** 2)

            # bias correction
            v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - beta1 ** t)
            v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - beta1 ** t)
            s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - beta2 ** t)
            s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - beta2 ** t)

            # update
            parameters["W" + str(l)] -= learning_rate * v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
            parameters["b" + str(l)] -= learning_rate * v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon)

        return parameters, v, s

    else:
        # standard gradient descent
        for l in range(1, L + 1):
            parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
            parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]

        return parameters


def create_mini_batches(X, Y, batch_size):
    """
    Splits the input data into mini-batches of the specified size.

    Parameters:
    -----------
    X : numpy.ndarray
        The input features of shape (m, n), where m is the number of examples 
        and n is the number of features.
    Y : numpy.ndarray
        The target labels of shape (m,) or (m, c), where m is the number of examples 
        and c is the number of classes (for one-hot encoding).
    batch_size : int
        The size of each mini-batch.

    Returns:
    --------
    list of tuples
        A list of mini-batches, where each mini-batch is a tuple (X_batch, Y_batch).
        X_batch is a numpy.ndarray of shape (batch_size, n) and Y_batch is a numpy.ndarray 
        of shape (batch_size,) or (batch_size, c). The last mini-batch may have fewer 
        examples if the total number of examples is not divisible by the batch size.

    Notes:
    ------
    - The data is shuffled before splitting into mini-batches to ensure randomness.
    - This function is commonly used in machine learning to prepare data for 
      stochastic gradient descent or mini-batch gradient descent.
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]

    mini_batches = []
    for k in range(0, m, batch_size):
        X_batch = X_shuffled[k:k + batch_size]
        Y_batch = Y_shuffled[k:k + batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches


def train_model(X_train, Y_train, X_val, Y_val, layers_dims, learning_rate=0.01, num_epochs=5000, print_every=1000, scheduling='none', decay_rate=0, min_lr=0,batch_size=0, beta1=0, beta2=0, epsilon=0, lambd=0, patience=0, keep_prob=1.0):
    def train_model(X_train, Y_train, X_val, Y_val, layers_dims, learning_rate=0.01, num_epochs=5000, print_every=1000, scheduling='none', decay_rate=0, min_lr=0, batch_size=0, beta1=0, beta2=0, epsilon=0, lambd=0, patience=0, keep_prob=1.0):
        """
        Trains a neural network model using gradient descent with optional features such as 
        learning rate scheduling, mini-batch gradient descent, Adam optimization, dropout, 
        L2 regularization, and early stopping.
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training input data of shape (number of features, number of training examples).
        Y_train : numpy.ndarray
            Training labels of shape (1, number of training examples).
        X_val : numpy.ndarray
            Validation input data of shape (number of features, number of validation examples).
        Y_val : numpy.ndarray
            Validation labels of shape (1, number of validation examples).
        layers_dims : list
            List containing the dimensions of each layer in the network.
        learning_rate : float, optional
            Initial learning rate for gradient descent (default is 0.01).
        num_epochs : int, optional
            Number of epochs to train the model (default is 5000).
        print_every : int, optional
            Frequency (in epochs) to print training and validation costs (default is 1000).
        scheduling : str, optional
            Type of learning rate scheduling ('none', 'linear', or 'exponential', default is 'none').
        decay_rate : float, optional
            Decay rate for learning rate scheduling (default is 0).
        min_lr : float, optional
            Minimum learning rate for linear scheduling (default is 0).
        batch_size : int, optional
            Size of mini-batches for gradient descent (default is 0, which means full batch).
        beta1 : float, optional
            Exponential decay hyperparameter for the first moment estimates in Adam (default is 0).
        beta2 : float, optional
            Exponential decay hyperparameter for the second moment estimates in Adam (default is 0).
        epsilon : float, optional
            Small value to prevent division by zero in Adam (default is 0).
        lambd : float, optional
            L2 regularization hyperparameter (default is 0).
        patience : int, optional
            Number of epochs with no improvement in validation cost before early stopping (default is 0).
        keep_prob : float, optional
            Probability of keeping a neuron active during dropout (default is 1.0, which means no dropout).
        Returns:
        --------
        parameters : dict
            Dictionary containing the trained parameters of the model.
        train_costs : list
            List of training costs recorded at each epoch.
        val_costs : list
            List of validation costs recorded at each epoch.
        Notes:
        ------
        - If `beta1`, `beta2`, and `epsilon` are non-zero, Adam optimization is used.
        - If `keep_prob` is less than 1.0, dropout is applied during training.
        - If `patience` is greater than 0, early stopping is enabled.
        - Learning rate scheduling can be linear or exponential based on the `scheduling` parameter.
        - Mini-batch gradient descent is used if `batch_size` is greater than 0.
        """
    use_adam = beta1 != 0 and beta2 != 0 and epsilon != 0
    use_dropout = keep_prob < 1.0
    
    parameters = initialize_parameters(layers_dims)

    if use_adam:
        v, s = initialize_adam(parameters)

    train_costs = []
    val_costs = []
    lrs = []

    best_val_cost = float('inf')
    best_parameters = None
    epochs_without_improvement = 0

    t = 1  # adam counter
    
    
    for epoch in tqdm(range(1, num_epochs + 1), desc="Training Epochs", ncols=100, unit="epoch"):
        # learning rate scheduling
        if scheduling == 'linear':
            current_lr = linear_decay(learning_rate, epoch, decay_rate=decay_rate, min_lr=min_lr)
        elif scheduling == 'exponential':
            current_lr = exponential_decay(learning_rate, epoch, decay_rate=decay_rate)
        else:
            current_lr = learning_rate

        lrs.append(current_lr)

        # mini-batch gradient descent
        if batch_size != 0:
            mini_batches = create_mini_batches(X_train, Y_train, batch_size)
            for X_batch, Y_batch in mini_batches:
                # forward propagation with or without dropout
                if use_dropout:
                    Y_hat, cache, dropout_masks = forward_propagation(X_batch, parameters, keep_prob=keep_prob)
                    grads = backward_propagation(parameters, cache, X_batch, Y_batch,
                                                lambd=lambd, dropout_masks=dropout_masks, keep_prob=keep_prob)
                else:
                    Y_hat, cache = forward_propagation(X_batch, parameters)
                    grads = backward_propagation(parameters, cache, X_batch, Y_batch, lambd=lambd)
                    
                if use_adam:
                    parameters, v, s = update_parameters(parameters, grads, current_lr, v, s, t, beta1, beta2, epsilon)
                else:
                    parameters = update_parameters(parameters, grads, current_lr)
                t += 1
        
        else:
            # full batch
            # Forward propagation with or without dropout
            if use_dropout:
                Y_hat, cache, dropout_masks = forward_propagation(X_train, parameters, keep_prob=keep_prob)
                grads = backward_propagation(parameters, cache, X_train, Y_train,
                                             lambd=lambd, dropout_masks=dropout_masks, keep_prob=keep_prob)
            else:
                Y_hat, cache = forward_propagation(X_train, parameters)
                grads = backward_propagation(parameters, cache, X_train, Y_train, lambd=lambd)

            if use_adam:
                parameters, v, s = update_parameters(parameters, grads, current_lr, v, s, t, beta1, beta2, epsilon)
            else:
                parameters = update_parameters(parameters, grads, current_lr)
            t += 1


        # costs
        Y_hat_train, _ = forward_propagation(X_train, parameters)
        cost_train = compute_cost(Y_hat_train, Y_train, parameters, lambd)
        train_costs.append(cost_train)

        # validation
        Y_hat_val, _ = forward_propagation(X_val, parameters)
        cost_val = compute_cost(Y_hat_val, Y_val, parameters, lambd)
        val_costs.append(cost_val)

        if print_every and epoch % print_every == 0:
            print(f"Epoch {epoch}: Train cost = {cost_train:.4f}, Val cost = {cost_val:.4f}")
        
        # early stopping
        if patience > 0:
            if cost_val < best_val_cost - 1e-4:
                best_val_cost = cost_val
                best_parameters = {k: v.copy() for k, v in parameters.items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch} (no improvement in {patience} epochs)")
                    parameters = best_parameters
                    break
    
    if scheduling != 'none':
        plt.plot(lrs)
        plt.title("Evolución del learning rate")
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.grid(True)
        plt.show()

    return parameters, train_costs, val_costs


def predict(X, parameters):
    """
    Perform predictions using the provided input data and model parameters.

    Args:
        X (numpy.ndarray): Input data of shape (number of features, number of examples).
        parameters (dict): Dictionary containing the model parameters (e.g., weights and biases).

    Returns:
        numpy.ndarray: The predicted output of the model, typically of shape (number of output units, number of examples).
    """
    AL, _ = forward_propagation(X, parameters)
    return AL


def one_hot_encode(y, num_classes=None):
    """
    Converts a vector of labels into a one-hot encoded matrix.

    Parameters:
    y (array-like): Array of integer labels to be one-hot encoded.
    num_classes (int, optional): The total number of classes. If not provided, 
                                  it is inferred as the maximum value in `y` plus 1.

    Returns:
    numpy.ndarray: A 2D array where each row corresponds to the one-hot encoded 
                   representation of the input label.
    """
    if num_classes is None:
        num_classes = np.max(y) + 1
    return np.eye(num_classes)[y]


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Plots a confusion matrix as a heatmap using the true and predicted labels.
    Args:
        y_true (array-like): Array of true class labels.
        y_pred (array-like): Array of predicted class labels.
        class_names (list, optional): List of class names to label the axes. 
            If None, numerical class indices will be used.
    Returns:
        None: Displays the confusion matrix plot.
    """
    num_classes = np.max(np.concatenate([y_true, y_pred])) + 1
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for t, p in zip(y_true, y_pred):
        conf_matrix[t, p] += 1

    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names is not None else np.arange(num_classes),
                yticklabels=class_names if class_names is not None else np.arange(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def report_metrics(X, y, parameters):
    """
    Evaluates the performance of a model by calculating accuracy, cross-entropy loss,
    and plotting the confusion matrix.
    Args:
        X (numpy.ndarray): Input data of shape (m, n), where m is the number of examples
            and n is the number of features.
        y (numpy.ndarray): True labels of shape (m, k), where k is the number of classes.
            If y is one-dimensional, it will be reshaped to (m, 1).
        parameters (dict): Model parameters used for prediction.
    Returns:
        tuple: A tuple containing:
            - accuracy (float): The accuracy of the predictions.
            - cross_entropy (float): The cross-entropy loss of the predictions.
    Side Effects:
        - Plots the confusion matrix comparing true labels and predicted labels.
    Notes:
        - The function assumes that `predict` and `compute_cost` are defined elsewhere
          in the codebase.
        - The input `y` and `y_pred` are expected to be one-hot encoded.
    """
    y_pred = predict(X, parameters)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
     
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y, axis=1)

    accuracy = np.mean(y_pred_labels == y_true_labels)
    cross_entropy = compute_cost(y_pred, y)
    
    plot_confusion_matrix(y_true_labels, y_pred_labels)
    
    return accuracy, cross_entropy


def report_train_validation_metrics(X_train, y_train, X_val, y_val, parameters=None, func=report_metrics,ret=False, model=None, time=None):
    """
    Reports training and validation metrics for a given model or set of parameters.

    This function calculates and displays the accuracy and cross-entropy metrics 
    for both the training and validation datasets. It supports two modes of operation:
    one where a model is provided, and another where parameters are passed to a 
    metric evaluation function.

    Args:
        X_train (array-like): Features of the training dataset.
        y_train (array-like): Labels of the training dataset.
        X_val (array-like): Features of the validation dataset.
        y_val (array-like): Labels of the validation dataset.
        parameters (dict, optional): Parameters to be passed to the metric evaluation 
            function. If None, the function assumes a model is provided. Defaults to None.
        func (callable, optional): Function to compute metrics. It should accept 
            different arguments depending on whether `parameters` is provided or not. 
            Defaults to `report_metrics`.
        ret (bool, optional): If True, returns the computed metrics. Defaults to False.
        model (object, optional): The model to evaluate. Required if `parameters` is None. 
            Defaults to None.
        time (float, optional): Time taken for training or evaluation, in minutes. 
            Defaults to None.

    Returns:
        tuple, optional: If `ret` is True, returns a tuple containing:
            - train_accuracy (float): Accuracy on the training dataset.
            - train_cross_entropy (float): Cross-entropy on the training dataset.
            - val_accuracy (float): Accuracy on the validation dataset.
            - val_cross_entropy (float): Cross-entropy on the validation dataset.

    Displays:
        - A DataFrame with training metrics (accuracy, cross-entropy, and time).
        - A DataFrame with validation metrics (accuracy, cross-entropy, and time).
    """
    if parameters is None:
        train_accuracy, train_cross_entropy = func(model, X_train, y_train)
        train_metrics_df = pd.DataFrame({
            'Accuracy': [f"{train_accuracy:.4f}"],
            'Cross-Entropy': [f"{train_cross_entropy:.4f}"],
            'Time (minutes)': [f"{time:.2f}"]
        })

        print("Métricas para el conjunto de entrenamiento:")
        display(train_metrics_df)

        val_accuracy, val_cross_entropy = func(model, X_val, y_val)
        val_metrics_df = pd.DataFrame({
            'Accuracy': [f"{val_accuracy:.4f}"],
            'Cross-Entropy': [f"{val_cross_entropy:.4f}"],
            'Time (minutes)': [f"{time:.2f}"]
        })

        print("Métricas para el conjunto de validación:")
        display(val_metrics_df)
        if ret:
            return train_accuracy, train_cross_entropy, val_accuracy, val_cross_entropy
    else:
        train_accuracy, train_cross_entropy = func(X_train, y_train, parameters)
        train_metrics_df = pd.DataFrame({
            'Accuracy': [f"{train_accuracy:.4f}"],
            'Cross-Entropy': [f"{train_cross_entropy:.4f}"],
            'Time (minutes)': [f"{time:.2f}"]
        })

        print("Métricas para el conjunto de entrenamiento:")
        display(train_metrics_df)

        val_accuracy, val_cross_entropy = func(X_val, y_val, parameters)
        val_metrics_df = pd.DataFrame({
            'Accuracy': [f"{val_accuracy:.4f}"],
            'Cross-Entropy': [f"{val_cross_entropy:.4f}"],
            'Time (minutes)': [f"{time:.2f}"]
        })

        print("Métricas para el conjunto de validación:")
        display(val_metrics_df)
        if ret:
            return train_accuracy, train_cross_entropy, val_accuracy, val_cross_entropy


def linear_decay(initial_lr, epoch, decay_rate=0.001, min_lr=1e-5):
    """
    Applies a linear decay to the learning rate.

    Parameters:
        initial_lr (float): The initial learning rate.
        epoch (int): The current epoch number.
        decay_rate (float, optional): The rate at which the learning rate decays per epoch. Default is 0.001.
        min_lr (float, optional): The minimum learning rate value. Default is 1e-5.

    Returns:
        float: The updated learning rate after applying linear decay, ensuring it does not go below `min_lr`.
    """
    return max(initial_lr - decay_rate * epoch, min_lr)


def exponential_decay(initial_lr, epoch, decay_rate=0.01):
    """
    Calculates the exponentially decayed learning rate.

    Parameters:
        initial_lr (float): The initial learning rate.
        epoch (int): The current epoch number.
        decay_rate (float, optional): The rate of decay. Defaults to 0.01.

    Returns:
        float: The updated learning rate after applying exponential decay.
    """
    return initial_lr * np.exp(-decay_rate * epoch)


def get_best_model(results_df):
    """
    Identifies and retrieves the best model from a DataFrame of model results based on the lowest 
    validation cross-entropy.

    Args:
        results_df (pd.DataFrame): A DataFrame containing model evaluation results. It must include 
                                   the following columns:
                                   - 'Validation Cross-Entropy': Cross-entropy loss on the validation set.
                                   - 'Arquitectura': The architecture of the model.
                                   - 'Hiperparámetros': The hyperparameters of the model.
                                   - 'Validation Accuracy': Accuracy on the validation set.

    Returns:
        dict: A dictionary containing details of the best model with the following keys:
              - "Arquitectura": The architecture of the best model.
              - "Hiperparámetros": The hyperparameters of the best model.
              - "Validation Accuracy": The validation accuracy of the best model.
              - "Validation Cross-Entropy": The validation cross-entropy of the best model.
    """
    best_model_idx = results_df['Validation Cross-Entropy'].idxmin()
    best_model = results_df.iloc[best_model_idx]

    return {
        "Arquitectura": best_model['Arquitectura'],
        "Hiperparámetros": best_model['Hiperparámetros'],
        "Validation Accuracy": best_model['Validation Accuracy'],
        "Validation Cross-Entropy": best_model['Validation Cross-Entropy']
    }


def predict_best_model(best_model, X_comp, parameters_M0=None, parameters_M1=None,
                       model_M2=None, model_M3=None, model_M4=None):
    """
    Predicts the output using the specified best model and its corresponding parameters or trained model.

    Parameters:
        best_model (str): The identifier of the best model to use for prediction. 
                            Must be one of "M0", "M1", "M2", "M3", or "M4".
        X_comp (array-like): The input data for which predictions are to be made.
        parameters_M0 (optional): Parameters for model "M0", if selected.
        parameters_M1 (optional): Parameters for model "M1", if selected.
        model_M2 (torch.nn.Module, optional): Trained PyTorch model for "M2", if selected.
        model_M3 (torch.nn.Module, optional): Trained PyTorch model for "M3", if selected.
        model_M4 (torch.nn.Module, optional): Trained PyTorch model for "M4", if selected.

    Returns:
        numpy.ndarray: The predicted probabilities or outputs for the input data.

    Raises:
        ValueError: If the specified `best_model` is not recognized.
    """
    if best_model == "M0":
        return predict(X_comp, parameters_M0)
    elif best_model == "M1":
        return predict(X_comp, parameters_M1)
    elif best_model == "M2":
        model_M2.eval()
        with torch.no_grad():
            logits = model_M2(torch.FloatTensor(X_comp))
            return F.softmax(logits, dim=1).numpy()
    elif best_model == "M3":
        model_M3.eval()
        with torch.no_grad():
            logits = model_M3(torch.FloatTensor(X_comp))
            return F.softmax(logits, dim=1).numpy()
    elif best_model == "M4":
        model_M4.eval()
        with torch.no_grad():
            logits = model_M4(torch.FloatTensor(X_comp))
            return F.softmax(logits, dim=1).numpy()
    else:
        raise ValueError(f"Model {best_model} not recognized")