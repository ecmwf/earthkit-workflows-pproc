import pytest
import numpy.random as random

from ppcascade import backends
from generic_tests import *


def random_array(*shape):
    return random.rand(*shape)


def to_numpy(array):
    return array


@pytest.fixture
def input_generator():
    return random_array


@pytest.fixture
def values():
    return to_numpy
