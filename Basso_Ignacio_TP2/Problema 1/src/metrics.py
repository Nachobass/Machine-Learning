import numpy as np
import matplotlib.pyplot as plt



def confusion_matrix(y_true, y_pred):
    """
    Computes the confusion matrix to evaluate the accuracy of a classification.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.

    Returns:
    --------
    cm : ndarray of shape (n_classes, n_classes)
        Confusion matrix where cm[i, j] represents the number of samples
        with true label being the i-th class and predicted label being the j-th class.

    Notes:
    ------
    - The classes are inferred from the unique values in `y_true`.
    - This implementation assumes that `y_true` and `y_pred` contain the same set of classes.
    """
    classes = np.unique(y_true)
    cm = np.zeros((len(classes), len(classes)), dtype=int)

    for i, actual in enumerate(classes):
        for j, predicted in enumerate(classes):
            cm[i, j] = np.sum((y_true == actual) & (y_pred == predicted))

    return cm


def accuracy(y_true, y_pred):
    """
    Calculate the accuracy of predictions.

    Accuracy is defined as the proportion of correctly predicted labels 
    to the total number of labels.

    Parameters:
    y_true (array-like): The true labels.
    y_pred (array-like): The predicted labels.

    Returns:
    float: The accuracy score, a value between 0 and 1, where 1 indicates 
           perfect accuracy and 0 indicates no correct predictions.
    """
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred, average='macro'):
    """
    Calculate the precision score for a classification task.
    Precision is the ratio of true positives to the sum of true positives and false positives.
    This function supports multi-class classification and calculates precision for each class.
    Args:
        y_true (array-like): Ground truth (true) labels.
        y_pred (array-like): Predicted labels.
        average (str, optional): Determines the type of averaging performed on the data.
            - 'macro': Calculate metrics for each class, and return their unweighted mean.
              This does not take class imbalance into account.
            - If not 'macro', the function will return a list of precision values for each class.
              Default is 'macro'.
    Returns:
        float or list: 
            - If `average='macro'`, returns the mean precision across all classes as a float.
            - Otherwise, returns a list of precision values for each class.
    Notes:
        - Precision is undefined when there are no predicted positives for a class. In such cases,
          the precision for that class is set to 0.
        - This implementation assumes `y_true` and `y_pred` are numpy arrays or can be converted
          to numpy arrays.
    """
    classes = np.unique(y_true)
    precisions = []
    
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        precisions.append(prec)
    
    return np.mean(precisions) if average == 'macro' else precisions


def recall(y_true, y_pred, average='macro'):
    """
    Compute the recall score for a multi-class classification problem.
    Recall is the ratio of correctly predicted positive observations to all 
    observations in the actual class. This implementation supports both 
    macro-averaged recall and class-wise recall.
    Parameters:
    -----------
    y_true : array-like
        Ground truth (true labels) for the dataset.
    y_pred : array-like
        Predicted labels for the dataset.
    average : str, optional, default='macro'
        Determines the type of averaging performed on the recall scores:
        - 'macro': Calculate metrics for each class, and return their unweighted mean.
        - If not 'macro', returns a list of recall scores for each class.
    Returns:
    --------
    float or list
        - If `average='macro'`, returns the macro-averaged recall as a float.
        - Otherwise, returns a list of recall scores for each class.
    Notes:
    ------
    - If a class has no true positive or false negative samples, its recall is set to 0.
    - This function assumes `y_true` and `y_pred` contain the same unique classes.
    """
    classes = np.unique(y_true)
    recalls = []
    
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        recalls.append(rec)
    
    return np.mean(recalls) if average == 'macro' else recalls


def f1_score(y_true, y_pred, average='macro'):
    """
    Calculate the F1 score, which is the harmonic mean of precision and recall.
    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated targets as returned by a classifier.
    average : str, optional, default='macro'
        Determines the type of averaging performed on the data:
        - 'macro': Calculate metrics for each label, and find their unweighted mean.
        - If not 'macro', the function will return a list of F1 scores for each label.
    Returns:
    --------
    float or list of floats
        The F1 score. If `average='macro'`, returns a single float representing the mean F1 score.
        Otherwise, returns a list of F1 scores for each label.
    Notes:
    ------
    - The F1 score is calculated as:
        F1 = 2 * (precision * recall) / (precision + recall)
      If both precision and recall are zero, the F1 score is set to 0 to avoid division by zero.
    - This implementation assumes that `precision` and `recall` functions are defined elsewhere
      and handle the computation of precision and recall respectively.
    """
    prec = precision(y_true, y_pred, average)
    rec = recall(y_true, y_pred, average)
    
    if isinstance(prec, list):
        f1 = [(2 * p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(prec, rec)]
        return np.mean(f1) if average == 'macro' else f1
    else:
        return (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0


def auc_roc(y_true, y_probs):
    """
    Compute the Area Under the Receiver Operating Characteristic Curve (AUC-ROC)
    for a binary classification task.
    The function calculates the True Positive Rate (TPR) and False Positive Rate (FPR)
    at various threshold levels and computes the AUC-ROC using the trapezoidal rule.
    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth (true binary labels), where each element is either 0 or 1.
    y_probs : array-like of shape (n_samples,)
        Predicted probabilities for the positive class, where each value is in the range [0, 1].
    Returns:
    --------
    auc_roc : float
        The computed AUC-ROC value, which ranges from 0 to 1. A value closer to 1 indicates
        better classification performance.
    Notes:
    ------
    - The function assumes that `y_true` contains binary labels (0 or 1).
    - The thresholds are evenly spaced between 0 and 1 with 500 points.
    - The AUC-ROC is computed using the trapezoidal rule on the sorted FPR and TPR values.
    """
    thresholds = np.linspace(0, 1, 500)
    tprs, fprs = [], []

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tprs.append(tpr)
        fprs.append(fpr)
    
    fprs = [0] + list(fprs) + [1]
    tprs = [0] + list(tprs) + [1]
    
    # Sort values ​​so that FPRS and Recalls are increasing
    fprs, tprs = zip(*sorted(zip(fprs, tprs)))
    auc_roc = np.trapz(tprs, fprs)
    return auc_roc


def auc_pr(y_true, y_probs):
    """
    Compute the Area Under the Precision-Recall Curve (AUC-PR).
    This function calculates the AUC-PR by iterating over a range of thresholds,
    computing precision and recall at each threshold, and then integrating the
    precision-recall curve using the trapezoidal rule.
    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth binary labels (0 or 1).
    y_probs : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    Returns:
    --------
    float
        The computed AUC-PR value.
    Notes:
    ------
    - The function assumes `y_true` contains binary values (0 or 1).
    - The `y_probs` array should contain predicted probabilities, not binary predictions.
    - The AUC-PR is calculated by sorting recall and precision values and integrating
      the precision-recall curve using the trapezoidal rule.
    """
    thresholds = np.linspace(0, 1, 500)
    precisions, recalls = [], []

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0

        precisions.append(prec)
        recalls.append(rec)

    # Sort values ​​so that FPRS and Recalls are increasing
    recalls, precisions = zip(*sorted(zip(recalls, precisions)))
    auc_pr = np.trapz(precisions, recalls)
    return auc_pr


def plot_curves(y_true, y_probs):
    """
    Plots the ROC (Receiver Operating Characteristic) curve and the Precision-Recall curve 
    for a binary classification model, and calculates the respective AUC (Area Under the Curve) values.
    Parameters:
    -----------
    y_true : array-like
        Ground truth (true binary labels), where each element is either 0 or 1.
    y_probs : array-like
        Predicted probabilities for the positive class (1), with values ranging from 0 to 1.
    Returns:
    --------
    tuple
        A tuple containing:
        - auc_roc (float): The Area Under the ROC Curve (AUC-ROC).
        - auc_pr (float): The Area Under the Precision-Recall Curve (AUC-PR).
    Notes:
    ------
    - The ROC curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) 
      at various threshold levels.
    - The Precision-Recall curve plots Precision against Recall at various threshold levels.
    - The AUC-ROC and AUC-PR values provide a single scalar metric to summarize the performance 
      of the classifier.
    - The function uses trapezoidal integration to compute the AUC values.
    """
    thresholds = np.linspace(0, 1, 500)
    tprs, fprs = [], []
    precisions, recalls = [], []

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0

        tprs.append(tpr)
        fprs.append(fpr)
        precisions.append(prec)
        recalls.append(rec)
    
    fprs = [0] + list(fprs) + [1]
    tprs = [0] + list(tprs) + [1]

    # Sort values ​​so that FPRS and Recalls are increasing
    fprs, tprs = zip(*sorted(zip(fprs, tprs)))
    recalls, precisions = zip(*sorted(zip(recalls, precisions)))

    auc_roc = np.trapz(tprs, fprs)
    auc_pr = np.trapz(precisions, recalls)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curve
    ax[0].plot(fprs, tprs, label=f'AUC-ROC = {auc_roc:.2f}')
    ax[0].plot([0, 1], [0, 1], 'k--')
    ax[0].set_xlabel("FPR (False Positive Rate)")
    ax[0].set_ylabel("TPR (True Positive Rate)")
    ax[0].set_title("ROC Curve")
    ax[0].legend()

    # Precision-Recall Curve
    ax[1].plot(recalls, precisions, label=f'AUC-PR = {auc_pr:.2f}')
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title("Precision-Recall Curve")
    ax[1].legend()

    plt.show()

    return auc_roc, auc_pr


def plot_multiple_curves(y_true, list_y_probs, model_names):
    """
    Plots ROC and Precision-Recall curves for multiple models.
    This function generates two subplots: one for the Receiver Operating Characteristic (ROC) curve
    and another for the Precision-Recall (PR) curve. It calculates the True Positive Rate (TPR),
    False Positive Rate (FPR), Precision, and Recall for a range of thresholds and computes the
    Area Under the Curve (AUC) for both ROC and PR curves.
    Args:
        y_true (array-like): Ground truth binary labels (0 or 1).
        list_y_probs (list of array-like): List of predicted probabilities for the positive class
            from different models. Each element in the list corresponds to a model.
        model_names (list of str): List of model names corresponding to the predicted probabilities
            in `list_y_probs`.
    Returns:
        None: The function displays the plots but does not return any value.
    Notes:
        - The ROC curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR).
        - The Precision-Recall curve plots Precision against Recall.
        - The AUC-ROC and AUC-PR values are displayed in the legend for each model.
        - The function assumes that `y_true` and each element in `list_y_probs` have the same length.
    """
    thresholds = np.linspace(0, 0.95, 500)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    for y_probs, label in zip(list_y_probs, model_names):
        tprs, fprs = [], []
        precisions, recalls = [], []

        for t in thresholds:
            y_pred = (y_probs >= t).astype(int)
            
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0

            tprs.append(tpr)
            fprs.append(fpr)
            precisions.append(prec)
            recalls.append(rec)

        fprs = [0] + fprs + [1]
        tprs = [0] + tprs + [1]
        fprs, tprs = zip(*sorted(zip(fprs, tprs)))
        recalls, precisions = zip(*sorted(zip(recalls, precisions)))

        auc_roc = np.trapz(tprs, fprs)
        auc_pr = np.trapz(precisions, recalls)

        ax[0].plot(fprs, tprs, label=f"{label} (AUC-ROC={auc_roc:.2f})")
        ax[1].plot(recalls, precisions, label=f"{label} (AUC-PR={auc_pr:.2f})")

    # ROC curve
    ax[0].plot([0, 1], [0, 1], 'k--')
    ax[0].set_xlabel("FPR")
    ax[0].set_ylabel("TPR")
    ax[0].set_title("Curva ROC")
    ax[0].legend()

    # PR curve
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title("Curva Precision-Recall")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


def evaluate_classification_performance(X, y, model, set_name='Validation'):
    """
    Evaluates the performance of a classification model on a given dataset.
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        The input features for the dataset.
    y : array-like of shape (n_samples,)
        The true labels for the dataset.
    model : object
        The trained classification model implementing `predict` and `predict_proba` methods.
    set_name : str, optional (default='Validation')
        The name of the dataset being evaluated (e.g., 'Validation', 'Test').
    Returns:
    --------
    None
        Prints the evaluation metrics including:
        - Confusion matrix
        - Accuracy
        - Precision
        - Recall
        - F1-score
        - AUC-ROC (Area Under the Receiver Operating Characteristic Curve)
        - AUC-PR (Area Under the Precision-Recall Curve)
    Notes:
    ------
    - If `model.predict` returns a probability matrix, the function converts it to class labels
      using `np.argmax`.
    - Assumes that `model.predict_proba` returns probabilities for all classes, and the second
      column corresponds to the positive class.
    - The function uses external utility functions such as `confusion_matrix`, `accuracy`,
      `precision`, `recall`, `f1_score`, and `plot_curves` to compute metrics and plot curves.
    """
    y_pred = model.predict(X)
    # If y_pred is a probability matrix, convert to class labels
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    y_probs = model.predict_proba(X)[:, 1]  # Probabilidades de la clase positiva
 
    cm = confusion_matrix(y, y_pred)
    acc = accuracy(y, y_pred)
    prec = precision(y, y_pred)
    rec = recall(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc_roc, auc_pr = plot_curves(y, y_probs)

    print(f"****Evaluación en {set_name}****")
    print("Matriz de confusión:\n", cm)
    print(f"✔ Accuracy: {acc:.2f}")
    print(f"✔ Precision: {prec:.2f}")
    print(f"✔ Recall: {rec:.2f}")
    print(f"✔ F1-score: {f1:.2f}")
    print(f"✔ AUC-ROC: {auc_roc:.2f}")
    print(f"✔ AUC-PR: {auc_pr:.2f}")