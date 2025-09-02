import numpy as np
import matplotlib.pyplot as plt
from collections import Counter



def f1_macro(y_true, y_pred):
    """
    Computes the macro-averaged F1 score for a multi-class classification problem.

    The macro-averaged F1 score is calculated by computing the F1 score for each class 
    independently and then taking the average of these scores. It treats all classes 
    equally, regardless of their frequency in the dataset.

    Parameters:
    -----------
    y_true : array-like
        Ground truth (true) labels for the dataset.
    y_pred : array-like
        Predicted labels for the dataset.

    Returns:
    --------
    float
        The macro-averaged F1 score across all classes. If a class has no true positives 
        or predicted positives, its F1 score is set to 0.

    Notes:
    ------
    - The F1 score is the harmonic mean of precision and recall, and it ranges from 0 to 1.
    - This implementation handles cases where there are no true positives or predicted 
      positives for a class by assigning an F1 score of 0 for that class.
    """
    classes = np.unique(y_true)
    f1s = []
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        if tp + fp == 0 or tp + fn == 0:
            f1 = 0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
        f1s.append(f1)
    return np.mean(f1s)


def compute_auc_pr(y_true, y_scores):
    """
    Compute the Area Under the Precision-Recall Curve (AUC-PR) for binary classification.

    This function calculates the AUC-PR by sorting the predicted scores in descending order
    and iteratively computing precision and recall values. The AUC-PR is then computed as
    the area under the precision-recall curve.

    Args:
        y_true (array-like): Ground truth binary labels (0 or 1).
        y_scores (array-like): Predicted scores or probabilities for the positive class.

    Returns:
        float: The computed AUC-PR value.

    Notes:
        - This implementation assumes that `y_true` and `y_scores` are NumPy arrays or
          array-like objects that can be indexed and sorted.
        - A small epsilon (1e-9) is added to the denominator when calculating recall to
          avoid division by zero.
    """
    desc_sort = np.argsort(-y_scores)
    y_true_sorted = y_true[desc_sort]
    precisions = []
    recalls = []
    tp = 0
    fp = 0
    fn = sum(y_true)
    for i in range(len(y_true)):
        if y_true_sorted[i] == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn + 1e-9)
        precisions.append(precision)
        recalls.append(recall)
    auc_pr = 0
    for i in range(1, len(precisions)):
        auc_pr += (recalls[i] - recalls[i - 1]) * precisions[i]
    return auc_pr


def compute_auc_roc(y_true, y_scores):
    """
    Compute the Area Under the Receiver Operating Characteristic Curve (AUC-ROC)
    for a set of true binary labels and predicted scores.

    Parameters:
    -----------
    y_true : array-like
        Array of true binary labels (0 or 1).
    y_scores : array-like
        Array of predicted scores or probabilities for the positive class.

    Returns:
    --------
    float
        The computed AUC-ROC value.

    Notes:
    ------
    - The function assumes that higher scores correspond to the positive class.
    - The AUC-ROC is calculated using the trapezoidal rule on the True Positive
      Rate (TPR) vs. False Positive Rate (FPR) curve.
    """
    desc_sort = np.argsort(-y_scores)
    y_true_sorted = y_true[desc_sort]
    tp = 0
    fp = 0
    tpr_list = []
    fpr_list = []
    P = sum(y_true)
    N = len(y_true) - P
    for i in range(len(y_true)):
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / P if P > 0 else 0
        fpr = fp / N if N > 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    auc_roc = 0
    for i in range(1, len(tpr_list)):
        auc_roc += (fpr_list[i] - fpr_list[i - 1]) * tpr_list[i]
    return auc_roc


def confusion_matrix(y_true, y_pred, n_classes):
    """
    Computes the confusion matrix for a classification task.
    Parameters:
    -----------
    y_true : list or array-like
        The true class labels.
    y_pred : list or array-like
        The predicted class labels.
    n_classes : int
        The number of classes in the classification task.
    Returns:
    --------
    numpy.ndarray
        A 2D array of shape (n_classes, n_classes) where the element at
        position (i, j) represents the number of instances of class i
        that were predicted as class j.
    Notes:
    ------
    - If a label in `y_true` or `y_pred` is outside the range [0, n_classes-1],
      a warning message will be printed, and the label will be ignored.
    - The matrix is initialized with zeros and populated based on the
      correspondence between `y_true` and `y_pred`.
    """
    mat = np.zeros((n_classes, n_classes), dtype=int)
    
    # Validate that the labels are within the allowed range
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            mat[t][p] += 1
        else:
            print(f"Advertencia: Etiqueta fuera de rango (t={t}, p={p})")
    return mat


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


def precision_recall_fscore(y_true, y_pred, n_classes):
    """
    Compute precision, recall, and F1-score for each class in a multi-class classification problem.

    Parameters:
        y_true (array-like): Ground truth (true) labels.
        y_pred (array-like): Predicted labels.
        n_classes (int): Number of classes in the classification problem.

    Returns:
        tuple: A tuple containing three lists:
            - precisions (list of float): Precision values for each class.
            - recalls (list of float): Recall values for each class.
            - f1s (list of float): F1-score values for each class.

    Notes:
        - Precision is the ratio of true positives to the sum of true positives and false positives.
        - Recall is the ratio of true positives to the sum of true positives and false negatives.
        - F1-score is the harmonic mean of precision and recall.
        - If the denominator for precision, recall, or F1-score is zero, the corresponding value is set to 0.
    """
    cm = confusion_matrix(y_true, y_pred, n_classes)
    precisions, recalls, f1s = [], [], []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return precisions, recalls, f1s


def roc_curve_multiclass(y_true, y_scores, n_classes):
    """
    Compute the Receiver Operating Characteristic (ROC) curve for a multiclass classification problem.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True class labels for the dataset. Each value should be an integer representing the class index.
    
    y_scores : array-like of shape (n_samples, n_classes)
        Predicted scores or probabilities for each class. Each row corresponds to a sample, and each column
        corresponds to the score for a specific class.
    
    n_classes : int
        The number of unique classes in the dataset.

    Returns:
    --------
    fpr_list : list of lists
        A list containing the false positive rates (FPR) for each class. Each element is a list of FPR values
        corresponding to different thresholds for the respective class.
    
    tpr_list : list of lists
        A list containing the true positive rates (TPR) for each class. Each element is a list of TPR values
        corresponding to different thresholds for the respective class.

    Notes:
    ------
    - The function computes the ROC curve for each class independently by treating it as a binary classification
      problem (one-vs-all approach).
    - The thresholds are determined based on the unique predicted scores for each class, sorted in descending order.
    - If there are no positive or negative samples for a class, the corresponding TPR or FPR values will be set to 0.
    """
    tpr_list, fpr_list = [], []
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        scores = y_scores[:, c]
        thresholds = np.unique(scores)[::-1]
        tpr, fpr = [], []
        for t in thresholds:
            pred_bin = (scores >= t).astype(int)
            tp = np.sum((pred_bin == 1) & (y_bin == 1))
            fp = np.sum((pred_bin == 1) & (y_bin == 0))
            fn = np.sum((pred_bin == 0) & (y_bin == 1))
            tn = np.sum((pred_bin == 0) & (y_bin == 0))
            tpr.append(tp / (tp + fn) if (tp + fn) else 0)
            fpr.append(fp / (fp + tn) if (fp + tn) else 0)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return fpr_list, tpr_list


def auc(x, y):
    """
    Compute the Area Under the Curve (AUC) using the trapezoidal rule.

    Parameters:
    x (array-like): The x-coordinates of the points defining the curve.
    y (array-like): The y-coordinates of the points defining the curve.

    Returns:
    float: The computed area under the curve.

    Notes:
    - The inputs `x` and `y` must have the same length.
    - The `x` values should be sorted in ascending order for accurate results.
    - This function uses `numpy.trapz` to perform the integration.
    """
    return np.trapz(y, x)


def plot_pr_roc_curves(y_true, y_probs, model_name, class_names=None):
    """
    Plots the Precision-Recall (PR) and Receiver Operating Characteristic (ROC) curves 
    for a multi-class classification model.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True class labels for the samples. Assumes integer labels for multi-class classification.
    
    y_probs : array-like of shape (n_samples, n_classes)
        Predicted probabilities for each class. Each row corresponds to the probability 
        distribution over all classes for a single sample.
    
    model_name : str
        Name of the model to be displayed in the plot titles.
    
    class_names : list of str, optional (default=None)
        List of class names corresponding to the class indices. If None, generic class 
        labels ("Clase 1", "Clase 2", etc.) will be used.

    Returns:
    --------
    None
        Displays the PR and ROC curves for each class in a single figure with two subplots.
        The ROC curve shows the trade-off between the True Positive Rate (TPR) and the 
        False Positive Rate (FPR), while the PR curve shows the trade-off between Precision 
        and Recall. Both curves include the Area Under the Curve (AUC) as part of the legend.
    """
    # Convert y_true to one-vs-rest format
    y_true_onehot = np.zeros_like(y_probs)
    y_true_onehot[np.arange(len(y_true)), y_true] = 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i in range(y_probs.shape[1]):
        # Get binarized labels for the current class
        y_bin = (y_true == i).astype(int)
        y_scores = y_probs[:, i]

        roc_auc = compute_auc_roc(y_bin, y_scores)
        pr_auc = compute_auc_pr(y_bin, y_scores)

        # Calculate points for the ROC and PR curves
        fpr, tpr = [], []
        thresholds = np.unique(y_scores)[::-1]
        for t in thresholds:
            pred_bin = (y_scores >= t).astype(int)
            tp = np.sum((pred_bin == 1) & (y_bin == 1))
            fp = np.sum((pred_bin == 1) & (y_bin == 0))
            fn = np.sum((pred_bin == 0) & (y_bin == 1))
            tn = np.sum((pred_bin == 0) & (y_bin == 0))
            tpr.append(tp / (tp + fn) if (tp + fn) else 0)
            fpr.append(fp / (fp + tn) if (fp + tn) else 0)

        precisions, recalls = [], []
        for t in thresholds:
            pred_bin = (y_scores >= t).astype(int)
            tp = np.sum((pred_bin == 1) & (y_bin == 1))
            fp = np.sum((pred_bin == 1) & (y_bin == 0))
            fn = np.sum((pred_bin == 0) & (y_bin == 1))
            precision = tp / (tp + fp) if tp + fp else 1
            recall = tp / (tp + fn) if tp + fn else 0
            precisions.append(precision)
            recalls.append(recall)

        # Label for the current class
        label = class_names[i] if class_names else f"Clase {i+1}"

        axes[0].plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.3f})')
        axes[1].plot(recalls, precisions, label=f'{label} (AUC = {pr_auc:.3f})')

    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].set_title(f'Curva ROC - {model_name}')
    axes[0].set_xlabel('FPR')
    axes[0].set_ylabel('TPR')
    axes[0].legend()

    axes[1].set_title(f'Curva Precision-Recall - {model_name}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def resumen_metricas(modelos, val_sets, n_classes):
    """
    Computes and summarizes various evaluation metrics for a set of models.

    Parameters:
    -----------
    modelos : dict
        A dictionary where keys are model names (str) and values are trained model objects.
    val_sets : dict
        A dictionary where keys are model names (str) and values are tuples (X_val, y_val),
        where X_val is the validation feature set (array-like) and y_val is the corresponding
        true labels (array-like).
    n_classes : int
        The number of classes in the classification problem.

    Returns:
    --------
    list
        A list of lists, where each inner list contains the following metrics for a model:
        - Model name (str)
        - Accuracy (float)
        - Average precision (float)
        - Average recall (float)
        - Average F1-score (float)
        - AUC-ROC (float or None if not applicable)
        - AUC-PR (float or None if not applicable)

    Notes:
    ------
    - If a model does not have a corresponding validation set in `val_sets`, it will be skipped.
    - If a model does not support probability predictions (e.g., `predict_proba`), AUC-ROC and
      AUC-PR metrics will not be computed for that model.
    - For models like Random Forest that do not directly support `predict_proba`, probabilities
      are estimated based on tree votes.
    - The function assumes the existence of helper functions `accuracy`, `precision_recall_fscore`,
      `roc_curve_multiclass`, and `compute_auc_pr` for metric calculations.
    """
    resumen = []

    for name, model in modelos.items():
        if name not in val_sets:
            print(f"No se encontró un set de validación para el modelo '{name}'. Saltando.")
            continue

        X_val, y_val = val_sets[name]
        y_pred = model.predict(X_val)

        acc = accuracy(y_val, y_pred)
        prec, rec, f1 = precision_recall_fscore(y_val, y_pred, n_classes)
        prec_avg = np.mean(prec)
        rec_avg = np.mean(rec)
        f1_avg = np.mean(f1)

        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_val)
        elif name == "Random Forest":
            votes = np.array([[tree.predict_one(x) for tree in model.trees] for x in X_val])
            probs = np.zeros((len(X_val), n_classes))
            for i in range(len(X_val)):
                vote_counts = Counter(votes[i])
                for cls, cnt in vote_counts.items():
                    probs[i, cls] = cnt / model.n_estimators
        else:
            print(f"El modelo {name} no soporta probabilidades. Saltando métricas AUC.")
            resumen.append([
                name, acc, prec_avg, rec_avg, f1_avg, None, None
            ])
            continue

        # AUC-ROC
        fprs, tprs = roc_curve_multiclass(y_val, probs, n_classes)
        auc_roc = np.mean([auc(fpr, tpr) for fpr, tpr in zip(fprs, tprs)])

        # AUC-PR
        precisiones, recalls = [], []
        for c in range(n_classes):
            y_bin = (y_val == c).astype(int)
            scores = probs[:, c]
            thresholds = np.unique(scores)[::-1]
            precs, recs = [], []
            for t in thresholds:
                pred_bin = (scores >= t).astype(int)
                tp = np.sum((pred_bin == 1) & (y_bin == 1))
                fp = np.sum((pred_bin == 1) & (y_bin == 0))
                fn = np.sum((pred_bin == 0) & (y_bin == 1))
                precision = tp / (tp + fp) if tp + fp else 1
                recall = tp / (tp + fn) if tp + fn else 0
                precs.append(precision)
                recs.append(recall)
            precisiones.append(precs)
            recalls.append(recs)

        auc_prs = [compute_auc_pr((y_val == c).astype(int), probs[:, c]) for c in range(n_classes)]
        auc_pr = np.mean(auc_prs)

        resumen.append([
            name, acc, prec_avg, rec_avg, f1_avg, auc_roc, auc_pr
        ])

    return resumen