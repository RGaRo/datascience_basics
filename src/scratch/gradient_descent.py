"""Module for gradient descent functions."""

from .linear_algebra import add, scalar_multiply
from .complex_typing import Vector


def gradient_step(v: Vector, gradient: Vector, step_size: float):
    """Moves step size in the gradient direction."""
    if len(v) != len(gradient):
        raise AssertionError(
            "Vector and gradient does't have same length"
        )
    else:
        step = scalar_multiply(step_size, gradient)
        return add(v, step)
