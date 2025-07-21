"""This is a function package for linear algebra related topics."""

import pytest
from typing import List


Vector = List[float]


def dot(v: Vector, w: Vector) -> float:
    """Computes v_1*w_1 + v_2*w_2 + ... + v_n*w_n"""
    if len(v) != len(w):
        raise AssertionError("Vectors must have same length")

    else:
        return sum([i * j for i, j in zip(v, w)])
