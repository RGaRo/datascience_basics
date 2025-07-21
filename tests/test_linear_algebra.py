"""Multiple tests for linear algebra module."""

import pytest
from scratch.linear_algebra import dot


def test_dot():
    assert dot([1, 2, 3], [1, 2, 3]) == 14
    assert dot([1, 1, 1], [1, 2, 3]) == 6
    with pytest.raises(
        AssertionError, match="Vectors must have same length"
    ):
        dot([1, 2, 3, 4], [1, 2, 3])
