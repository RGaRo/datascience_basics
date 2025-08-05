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
