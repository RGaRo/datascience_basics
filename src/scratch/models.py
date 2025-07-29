"""Module for machine learning models."""
# Third party libraries
import numpy as np
# Specific functions
from math import log, exp
from statistics import mean
from collections import Counter, defaultdict
from sklearn.metrics import confusion_matrix
from typing import List, Dict, Iterable, Tuple, Literal
# Own library
from .complex_typing import Vector
from .linear_algebra import distance
from .metrics import precision, recall, f1_score
from .data_preprocessing import LabeledPoint, Message, tokenize


class KNNClassifier:
    """A class to represent a KNN model."""

    def __init__(self, k: int = 5):
        """
        k: How many neighbours you want consider
        """
        self.labeled_points: List[LabeledPoint] = []
        self.k = k

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
    def _cm(self, test_dataset: List[LabeledPoint]) -> Dict[str, float]:
        """ Get the confusion matrix of each label
            Cols - Predicted
            Rows - Actual
        """
        real_labels = [lp.label for lp in test_dataset]
        predicted_labels = [self.predict(lp.point) for lp in test_dataset]
        labels = sorted(list(set(real_labels + predicted_labels)), reverse=True)
        cm = confusion_matrix(real_labels, predicted_labels, labels=labels)
        
        cm_detailed = defaultdict(dict)
        for label_index in range(len(labels)):
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for row in range(len(labels)):
                for col in range(len(labels)):
                    if row==label_index and col==label_index:
                        tp += int(cm[row][col])
                    elif row==label_index and col!=label_index:
                        fn += int(cm[row][col])
                    elif row!=label_index and col==label_index:
                        fp += int(cm[row][col])
                    elif row!=label_index and col!=label_index:
                        tn += int(cm[row][col])

            cm_detailed[labels[label_index]]['tp'] = tp
            cm_detailed[labels[label_index]]['tn'] = tn
            cm_detailed[labels[label_index]]['fp'] = fp
            cm_detailed[labels[label_index]]['fn'] = fn

        return labels, cm, cm_detailed
    
    def metrics(self ,test_dataset: List[LabeledPoint], kind: Literal['micro','macro'] = 'micro') -> Dict[str, float]:
        labels, cm, cm_detailed = self._cm(test_dataset)
        # If just two labels we get the simpler confusion matrix and got metrics
        if len(labels) == 2:
            tp = cm_detailed[labels[0]]['tp']
            fp = cm_detailed[labels[0]]['fp']
            fn = cm_detailed[labels[0]]['fn']
            tn = cm_detailed[labels[0]]['tn']
            _accuracy = float(np.trace(cm)/np.sum(cm))
            _precision = precision(tp, fp)
            _recall = recall(tp, fn)
            _f1_score = f1_score(tp, tn, fp, fn)

            return {
                "labels": labels,
                "confusion_matrix": cm,
                "confusion_matrix_detailes": cm_detailed,
                "accuracy": _accuracy, 
                "precision": _precision, 
                "recall": _recall,
                "f1_score": _f1_score
            }
        
        # If multilabeled predictions we got micro or macro metrics
        if len(labels) > 2:
            _accuracy = float(np.trace(cm)/np.sum(cm))
            if kind == 'micro':
                tp = sum([cm_detailed[label]['tp'] for label in labels])
                fp = sum([cm_detailed[label]['fp'] for label in labels])
                fn = sum([cm_detailed[label]['fn'] for label in labels])
                tn = sum([cm_detailed[label]['tn'] for label in labels])
                _precision = precision(tp, fp)
                _recall = recall(tp, fn)
                _f1_score = f1_score(tp, tn, fp, fn)

            elif kind == 'macro':
                _precision = mean(
                    [
                        precision(
                            cm_detailed[label]['tp'],
                            cm_detailed[label]['fp']
                        )
                        for label in labels
                    ]
                
                )

                _recall = mean(
                    [
                        recall(
                            cm_detailed[label]['tp'],
                            cm_detailed[label]['fn']
                        )
                        for label in labels
                    ]
                
                )

                _f1_score = mean(
                    [
                        f1_score(
                            cm_detailed[label]['tp'],
                            cm_detailed[label]['tn'],
                            cm_detailed[label]['fp'],
                            cm_detailed[label]['fn']
                        )
                        for label in labels
                    ]
                
                )

            else:
                raise AssertionError('Not a valid kind of metrics, use kind = "micro" or kind = "macro"')
            
            return {
                "labels": labels,
                "confusion_matrix": cm,
                "confusion_matrix_detailes": cm_detailed,
                "accuracy": _accuracy, 
                "precision": _precision, 
                "recall": _recall,
                "f1_score": _f1_score
            }

    
class NaiveBayesSpamClassifier:
    """
    A model to predict if a message is spam where
    k = smothing value
    """
    def __init__(self, k: float = 0.5) -> None:
        self.k = k
        self.tokens = set()
        self.token_spam_count: Dict[str, int] = defaultdict(int)
        self.token_ham_count: Dict[str, int] = defaultdict(int)
        self.spam_messages = 0
        self.ham_messages = 0
    
    def fit(self, messages: Iterable[Message]) -> None:
        for message in messages:
            if message.is_spam:
                self.spam_messages += 1
            elif not message.is_spam:
                self.ham_messages += 1
        
            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_count[token] += 1
                elif not message.is_spam:
                    self.token_ham_count[token] += 1
    
    def _probabilities(self, token: str) -> Tuple[float, float, float, float]:
        """
        Computes the probabilities listed below:
        P(token|spam)
        P(¬token|spam)
        P(token|ham)
        P(¬token|ham)
        """
        p_token_given_spam = (self.k + self.token_spam_count[token])/ ((2*self.k) + (self.spam_messages))
        p_not_token_given_spam = 1 - p_token_given_spam
        p_token_given_ham = (self.k + self.token_ham_count[token])/ ((2*self.k) + (self.ham_messages))
        p_not_token_given_ham = 1 - p_token_given_ham

        return p_token_given_spam, p_not_token_given_spam, p_token_given_ham, p_not_token_given_ham
    
    def predict(self, text: str) -> float:
        """
        Return the probability of spam given the tokens.
        """
        text_token = tokenize(text)
        log_prob_token_given_spam = 0
        log_prob_token_given_ham = 0

        # Iterate through each token
        for token in self.tokens:
            p_token_given_spam, p_not_token_given_spam, p_token_given_ham, p_not_token_given_ham = self._probabilities(token)
            
            if token in text_token:
                log_prob_token_given_spam += log(p_token_given_spam)
                log_prob_token_given_ham += log(p_token_given_ham)
            
            elif token not in text_token:
                log_prob_token_given_spam += log(p_not_token_given_spam)
                log_prob_token_given_ham += log(p_not_token_given_ham)
            
        p_tokens_given_spam = exp(log_prob_token_given_spam)
        p_tokens_given_ham = exp(log_prob_token_given_ham)

        return p_tokens_given_spam/(p_tokens_given_spam + p_tokens_given_ham)
    
    def _cm(self, test_messages: List[Message], threshold: float = 0.5) -> Dict[str, float]:
        """ Get the confusion matrix of each label
            Cols - Predicted
            Rows - Actual
        """
        real_labels = [message.is_spam for message in test_messages]
        predicted_labels = [self.predict(message.text)>threshold for message in test_messages]
        labels = sorted(list(set(real_labels + predicted_labels)), reverse=True)
        cm = confusion_matrix(real_labels, predicted_labels, labels=labels)
        
        cm_detailed = defaultdict(dict)
        for label_index in range(len(labels)):
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for row in range(len(labels)):
                for col in range(len(labels)):
                    if row==label_index and col==label_index:
                        tp += int(cm[row][col])
                    elif row==label_index and col!=label_index:
                        fn += int(cm[row][col])
                    elif row!=label_index and col==label_index:
                        fp += int(cm[row][col])
                    elif row!=label_index and col!=label_index:
                        tn += int(cm[row][col])

            cm_detailed[labels[label_index]]['tp'] = tp
            cm_detailed[labels[label_index]]['tn'] = tn
            cm_detailed[labels[label_index]]['fp'] = fp
            cm_detailed[labels[label_index]]['fn'] = fn

        return labels, cm, cm_detailed
    
    def metrics(self ,test_dataset: List[LabeledPoint], 
                threshold: float = 0.5 ,
                kind: Literal['micro','macro'] = 'micro') -> Dict[str, float]:
        labels, cm, cm_detailed = self._cm(test_dataset, threshold = threshold)

        if len(labels) == 2:
            tp = cm_detailed[labels[0]]['tp']
            fp = cm_detailed[labels[0]]['fp']
            fn = cm_detailed[labels[0]]['fn']
            tn = cm_detailed[labels[0]]['tn']
            _accuracy = float(np.trace(cm)/np.sum(cm))
            _precision = precision(tp, fp)
            _recall = recall(tp, fn)
            _f1_score = f1_score(tp, tn, fp, fn)
            
            return {
                "labels": labels,
                "confusion_matrix": cm,
                "confusion_matrix_detailes": cm_detailed,
                "accuracy": _accuracy, 
                "precision": _precision, 
                "recall": _recall,
                "f1_score": _f1_score
            }

        elif len(labels) > 2:
            _accuracy = float(np.trace(cm)/np.sum(cm))
            if kind == 'micro':
                tp = sum([cm_detailed[label]['tp'] for label in labels])
                fp = sum([cm_detailed[label]['fp'] for label in labels])
                fn = sum([cm_detailed[label]['fn'] for label in labels])
                tn = sum([cm_detailed[label]['tn'] for label in labels])
                _precision = precision(tp, fp)
                _recall = recall(tp, fn)
                _f1_score = f1_score(tp, tn, fp, fn)

            elif kind == 'macro':
                _precision = mean(
                    [
                        precision(
                            cm_detailed[label]['tp'],
                            cm_detailed[label]['fp']
                        )
                        for label in labels
                    ]
                
                )

                _recall = mean(
                    [
                        recall(
                            cm_detailed[label]['tp'],
                            cm_detailed[label]['fn']
                        )
                        for label in labels
                    ]
                
                )

                _f1_score = mean(
                    [
                        f1_score(
                            cm_detailed[label]['tp'],
                            cm_detailed[label]['tn'],
                            cm_detailed[label]['fp'],
                            cm_detailed[label]['fn']
                        )
                        for label in labels
                    ]
                )

            else:
                raise AssertionError('Not a valid kind of metrics, use kind = "micro" or kind = "macro"')
            
            return {
                "labels": labels,
                "confusion_matrix": cm,
                "confusion_matrix_detailes": cm_detailed,
                "accuracy": _accuracy, 
                "precision": _precision, 
                "recall": _recall,
                "f1_score": _f1_score
            }