"""Multiple tests for data preprocessing module."""

import pytest
from scratch.data_preprocessing import split_data, train_test_split


def test_split_data():
    ls_1, ls_2 = split_data([1, 2, 3, 4, 5], 0.50)
    ls_3, ls_4 = split_data([i for i in range(1000)], 0.80)
    assert (
        len(ls_1) == 2
        and len(ls_2) == 3
        and len(ls_3) == 800
        and len(ls_4) == 200
    )


def test_train_test_split():
    x_train, y_train, x_test, y_test = train_test_split(
        [i for i in range(1000)], [i**2 for i in range(1000)], 0.80
    )
    assert (
        len(x_train) == len(y_train) == 800
        and len(x_test) == len(y_test) == 200
    )
    assert sum([x**2 for x in x_train]) == sum(y_train)
    assert sum([x**2 for x in x_test]) == sum(y_test)
    with pytest.raises(
        AssertionError,
        match="x_vals vector and y_vals vector must have same length",
    ):
        train_test_split([1, 2, 3, 4], [1, 2, 3], 0.5)
