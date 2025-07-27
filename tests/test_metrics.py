"""Multiple tests for model metrics module."""

import pytest
from scratch.metrics import accuracy, precision, recall, f1_score


@pytest.mark.parametrize(
    "tp, tn, fp, fn, expected",
    [
        (70, 981070, 4930, 13930, 0.98114),
    ],
)
def test_accuracy(tp, tn, fp, fn, expected):
    assert accuracy(tp, tn, fp, fn) == expected


@pytest.mark.parametrize(
    "tp, tn, fp, fn, expected",
    [
        (70, 981070, 4930, 13930, 0.014),
    ],
)
def test_precision(tp, tn, fp, fn, expected):
    assert precision(tp, fp) == expected


@pytest.mark.parametrize(
    "tp, tn, fp, fn, expected",
    [
        (70, 981070, 4930, 13930, 0.005),
    ],
)
def test_recall(tp, tn, fp, fn, expected):
    assert recall(tp, fn) == expected


@pytest.mark.parametrize(
    "tp, tn, fp, fn, expected",
    [
        (70, 981070, 4930, 13930, 0.007368),
    ],
)
def test_f1_score(tp, tn, fp, fn, expected):
    assert round(f1_score(tp, tn, fp, fn), 5) == round(expected, 5)
