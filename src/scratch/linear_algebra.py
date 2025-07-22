"""This is a function package for linear algebra related topics."""

import math
import pytest
from typing import List
from scratch.complex_typing import Vector


# Dot product of two vectors
def dot(v: Vector, w: Vector) -> float:
    """Computes v_1*w_1 + v_2*w_2 + ... + v_n*w_n"""
    if len(v) != len(w):
        raise AssertionError("Vectors must have same length")

    else:
        return sum([i * j for i, j in zip(v, w)])


# Multply a vector for a scalar
def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiply every element for c"""
    return [c * element for element in v]


# Substraction of two vectors of the same size
def substract(v: Vector, w: Vector) -> Vector:
    """Substract corresponding elements"""
    if len(v) != len(w):
        raise AssertionError("Vectors must have same length")
    else:
        return [v_i - w_i for (v_i, w_i) in zip(v, w)]


# Addition of two vectors of the same size
def add(v: Vector, w: Vector) -> Vector:
    """Addition of corresponding elements"""
    if len(v) != len(w):
        raise AssertionError("Vectors must have same length")
    else:
        return [v_i + w_i for (v_i, w_i) in zip(v, w)]


# Calculate sum of squares
def sum_of_squares(v: Vector) -> float:
    """Returns v_1*v_1 + v_2*v_2 + ... + v_n*v_n"""
    return dot(v, v)


# Calculate the magnitude
def magnitude(v: Vector) -> float:
    "Return the length of the vector"
    return math.sqrt(sum_of_squares(v))


# Calculate distance between 2 vectors
def distance(v: Vector, w: Vector) -> float:
    """Retruns the distance between 2 vectors"""
    return magnitude(substract(v, w))


# Calculate the mean
def vector_mean(vectors: List[Vector]) -> Vector:
    """Compute the element wise average"""
    assert all(
        len(v) for v in vectors
    ), "Not all vectors have the same size"
    vectors_qty = len(vectors)
    return [
        sum(v[i] for v in vectors) / vectors_qty
        for i in range(len(vectors[0]))
    ]
