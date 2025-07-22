"""This is a module for complex typing variables."""

from typing import List, TypeVar

Vector = List[float]
Matrix = List[List[float]]
T = TypeVar("T")
X = TypeVar("X")
Y = TypeVar("Y")
Stat = TypeVar("Stat")
X_TEST = TypeVar("X_TEST")
X_TRAIN = TypeVar("X_TRAIN")
Y_TEST = TypeVar("Y_TEST")
Y_TRAIN = TypeVar("Y_TRAIN")
