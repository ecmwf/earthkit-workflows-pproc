import pytest
import numpy.random as random

from earthkit.data import FieldList
from earthkit.data.core.metadata import RawMetadata

from ppcascade import backends
from generic_tests import *


class MockMetaData(RawMetadata):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def buffer_to_metadata(self) -> "MockMetaData":
        return self


def random_fieldlist(*shape) -> FieldList:
    return FieldList.from_numpy(
        random.rand(*shape), [MockMetaData() for x in range(shape[0])]
    )


def to_numpy(fl: FieldList):
    return fl.values


@pytest.fixture
def input_generator():
    return random_fieldlist


@pytest.fixture
def values():
    return to_numpy


@pytest.mark.parametrize(
    "func",
    [
        backends.mean,
        backends.std,
        backends.max,
        backends.min,
        backends.prod,
        backends.sum,
        backends.norm,
    ],
)
def test_multi_arg(func, input_generator, values):
    arr = [input_generator(2, 20), input_generator(2, 20), input_generator(1, 20)]
    concat = backends.concat(*arr)
    assert len(concat) == 5
    assert values(concat).shape == (5, 20)
    nested = func(concat)
    assert values(nested).shape == (1, 20)

    with pytest.raises(ValueError):
        func(*arr)

    # Field lists with multiple fields
    arr2 = [input_generator(3, 20) for _ in range(5)]
    assert values(func(*arr2)).shape == (3, 20)


@pytest.mark.parametrize(
    "func",
    [
        backends.add,
        backends.subtract,
        backends.multiply,
        backends.divide,
        backends.diff,
    ],
)
def test_two_arg(input_generator, values, func):
    # Single field in each field list
    arr = [input_generator(1, 5) for _ in range(2)]
    assert values(func(*arr)).shape == (1, 5)

    # Multiple fields in each field list
    arr = [input_generator(3, 5), 2]
    unnested = func(*arr)
    assert values(unnested).shape == (3, 5)

    # Raises on too many arguments
    arr = [input_generator(1, 5) for _ in range(3)]
    with pytest.raises(AssertionError):
        func(*arr)
    with pytest.raises(AssertionError):
        func(backends.concat(*arr))


def test_concat(input_generator, values):
    arr = [input_generator(1, 5) for _ in range(3)]
    res = backends.concat(*arr)
    assert len(res) == 3
    assert np.all(values(res[0]) == values(arr[0]))
    assert np.all(values(res[-1]) == values(arr[-1]))


@pytest.mark.parametrize(
    ["args", "kwargs", "output_shape"],
    [
        [[0], {"axis": 0}, (1, 2, 3)],
        [[[0]], {"axis": 0}, (1, 2, 3)],
        [[[0, 1]], {"axis": 0}, (2, 2, 3)],
    ],
)
def test_take(input_generator, values, args, kwargs, output_shape):
    input = input_generator(3, 2, 3)
    output = backends.take(input, *args, **kwargs)
    assert values(output).shape == output_shape
