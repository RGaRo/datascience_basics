"""Module for machine learning models."""

from typing import NamedTuple, List
from collections import Counter
from .linear_algebra import distance
from .complex_typing import Vector
from .metrics import accuracy, precision, recall, f1_score


class LabeledPoint(NamedTuple):
    point: Vector
    label: str


class KNNClassifier:
    """A class to represent a KNN model."""

    def __init__(self, k: int = 5):
        """
        k: How many neighbours you want consider
        """
        self.labeled_points: List[LabeledPoint] = []
        self.k = k
        self.val_metrics = dict()

    def fit(self, labeled_points: List[LabeledPoint]):
        """Store labeled points for KNN classification."""
        self.labeled_points = labeled_points

    def predict(self, new_point: Vector):
        """Predict the label of a new point.
        new_point: Vector you want to predict
        """
        # We got the labeled pointes ordered by distance
        ordered_labeled_points = sorted(
            self.labeled_points,
            key=lambda x: distance(x.point, new_point),
        )
        ordered_labels = [
            labeled_point.label
            for labeled_point in ordered_labeled_points
        ]

        tiebreaker_counter = 0
        winners_count = 0
        while winners_count != 1:
            # Getting the labels in order
            k_nearest_labels = [
                label
                for label in ordered_labels[
                    : self.k - tiebreaker_counter
                ]
            ]

            # Define the winner label
            k_nearest_labels_counter = Counter(k_nearest_labels)
            frecuencies = [i for i in k_nearest_labels_counter.values()]
            max_frecuencies = max(frecuencies)
            winners_count = len(
                [i for i in frecuencies if i == max_frecuencies]
            )

            tiebreaker_counter += 1
        return k_nearest_labels_counter.most_common(1)[0][0]

    def metrics(self, test_labeled_points: List[LabeledPoint]):
        """Computes metrics vs a validation dataset."""
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        test_labels = sorted(
            set(
                [
                    test_labeled_point.label
                    for test_labeled_point in test_labeled_points
                ]
            )
        )

        test_predictions = [
            (
                test_labeled_point.label,
                self.predict(test_labeled_point.point),
            )
            for test_labeled_point in test_labeled_points
        ]

        predictions_encoded = [
            [
                [int(real == label) for label in test_labels],
                [int(pred == label) for label in test_labels],
            ]
            for real, pred in test_predictions
        ]

        for real_encoded, prediction_encoded in predictions_encoded:
            for i in range(len(test_labels)):
                if real_encoded[i] == 1 and prediction_encoded[i] == 1:
                    tp += 1
                elif (
                    real_encoded[i] == 0 and prediction_encoded[i] == 1
                ):
                    fp += 1
                elif (
                    real_encoded[i] == 1 and prediction_encoded[i] == 0
                ):
                    fn += 1
                elif (
                    real_encoded[i] == 0 and prediction_encoded[i] == 0
                ):
                    tn += 1

        accuracy_ = accuracy(tp, tn, fp, fn)
        precision_ = precision(tp, fp)
        recall_ = recall(tp, fn)
        f1_score_ = f1_score(tp, tn, fp, fn)

        self.val_metrics = {
            "Accuracy": accuracy_,
            "Precision": precision_,
            "Recall": recall_,
            "F1 Score": f1_score_,
        }
