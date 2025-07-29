"""Module for data preprocessing."""

import random
import re
from typing import Tuple, List, NamedTuple, Set
from .complex_typing import X, Y, Vector


class LabeledPoint(NamedTuple):
    point: Vector
    label: str


class Message(NamedTuple):
    text: str
    is_spam: bool


def split_data(
    data: List[X], proportion: float
) -> Tuple[List[X], List[X]]:
    """Split data into fractions"""
    data_list = data[:]
    random.shuffle(data_list)
    slide_limit = int(len(data_list) * proportion)
    return data_list[:slide_limit], data_list[slide_limit:]


def train_test_split(
    x_vals: List[X], y_vals: List[Y], test_proportion: float
) -> Tuple[List[X], List[Y], List[X], List[Y]]:
    """Function to make a split into train and test dataset."""
    if len(x_vals) != len(y_vals):
        raise AssertionError(
            "x_vals vector and y_vals vector must have same length"
        )
    else:
        idxs = [i for i in range(len(x_vals))]
        train_idxs, test_idxs = split_data(idxs, test_proportion)
        return (
            [x_vals[i] for i in train_idxs],
            [y_vals[i] for i in train_idxs],
            [x_vals[i] for i in test_idxs],
            [y_vals[i] for i in test_idxs],
        )


def tokenize(text: str) -> Set[str]:
    """Reduce a text into a set of its words."""
    text = text.lower()
    all_words = re.findall("[a-z0-9]+", text)
    return set(all_words)


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
