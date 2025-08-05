"""Module for model metrics."""


def accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    """Compute the accuracy taking
    tp: true positives
    tn: true negatives
    fp: false positives
    fn: false negatives
    as inputs
    """
    return (tp + tn) / (tp + tn + fp + fn)


def precision(tp: int, fp: int) -> float:
    """Compute the precision taking
    tp: true positives
    fp: false positives
    as inputs

    Percentage of correctly predicted positive instances
    out of all instances predicted as positive.
    """
    return (tp) / (tp + fp)


def recall(tp: int, fn: int) -> float:
    """Compute the recall taking
    tp: true positives
    fn: false negativ
    as inputs

    The percentage of true positive predictions
    out of all actual positive cases.
    """
    return (tp) / (tp + fn)


def f1_score(tp: int, tn: int, fp: int, fn: int) -> float:
    """Compute the accuracy taking
    tp: true positives
    tn: true negatives
    fp: false positives
    fn: false negatives
    as inputs
    The F1 Score is the harmonic mean of precision and recall,
    providing a balance between the two when there is
    an uneven class distribution.
    """
    p = precision(tp, fp)
    r = recall(tp, fn)
    return 2 * p * r / (p + r)


def mean_squared_error(y_true, y_pred):
    """
    Computes the MSE of 2 lists
    """
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(
        y_true
    )


def print_classification_report(report: dict):
    labels = report["labels"]
    cm = report["confusion_matrix"]
    cm_details = report["confusion_matrix_detailes"]

    print("\n=== Classification Report ===")
    print(f"{'Labels:':<5} {labels}")

    print("\nConfusion Matrix:")
    for row in cm:
        print("  " + "  ".join(f"{val:>2}" for val in row))

    print("\nDetailed Confusion Metrics per Class:")
    for label in labels:
        metrics = cm_details[label]
        print(
            f"  {label:<12} -> "
            f"TP: {metrics['tp']:>2}, "
            f"TN: {metrics['tn']:>2}, "
            f"FP: {metrics['fp']:>2}, "
            f"FN: {metrics['fn']:>2}"
        )

    print("\nOverall Metrics:")
    print(f"  Accuracy : {report['accuracy']:.4f}")
    print(f"  Precision: {report['precision']:.4f}")
    print(f"  Recall   : {report['recall']:.4f}")
    print(f"  F1 Score : {report['f1_score']:.4f}")
