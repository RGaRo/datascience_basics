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
