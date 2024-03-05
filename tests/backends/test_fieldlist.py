import pytest
import numpy.random as random
import numpy as np

from earthkit.data import FieldList
from earthkit.data.core.metadata import RawMetadata

from ppcascade.backends.fieldlist import NumpyFieldListBackend
from ppcascade.utils.window import Range


class MockMetaData(RawMetadata):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def buffer_to_metadata(self) -> "MockMetaData":
        return self


def random_fieldlist(*shape) -> FieldList:
    return FieldList.from_numpy(
        random.rand(*shape), [MockMetaData() for x in range(shape[0])]
    )


def test_instantiation():
    NumpyFieldListBackend()


@pytest.mark.parametrize(
    "func",
    [
        NumpyFieldListBackend.mean,
        NumpyFieldListBackend.std,
        NumpyFieldListBackend.max,
        NumpyFieldListBackend.min,
        NumpyFieldListBackend.prod,
        NumpyFieldListBackend.sum,
    ],
)
def test_multi_arg(func):
    arr = [random_fieldlist(1, 5) for _ in range(5)]
    unnested = func(*arr)
    nested = func(NumpyFieldListBackend.concat(*arr))
    assert np.all(unnested.values == nested.values)


@pytest.mark.parametrize(
    "func",
    [
        NumpyFieldListBackend.add,
        NumpyFieldListBackend.subtract,
        NumpyFieldListBackend.multiply,
        NumpyFieldListBackend.divide,
        NumpyFieldListBackend.norm,
        NumpyFieldListBackend.diff,
    ],
)
def test_two_arg(func):
    arr = [random_fieldlist(1, 5) for _ in range(2)]
    unnested = func(*arr)
    nested = func(NumpyFieldListBackend.concat(*arr))
    assert np.all(unnested.values == nested.values)


@pytest.mark.parametrize(
    "func",
    [
        NumpyFieldListBackend.divide,
        NumpyFieldListBackend.norm,
    ],
)
def test_two_arg(func):
    arr = [random_fieldlist(1, 5) for _ in range(3)]
    with pytest.raises(AssertionError):
        func(*arr)
    with pytest.raises(AssertionError):
        func(NumpyFieldListBackend.concat(*arr))


@pytest.mark.parametrize(
    "comparison",
    [
        "<=",
        "<",
        ">=",
        ">",
    ],
)
def test_threshold(comparison):
    config = {"comparison": comparison, "value": 2}
    NumpyFieldListBackend.threshold(random_fieldlist(1, 5), **config)


def test_extreme():
    ens = random_fieldlist(5, 5)
    ens.metadata()[0]._d.update(
        {
            "timeRangeIndicator": 3,
            "date": "0",
            "subCentre": 0,
            "totalNumber": 0,
        }
    )
    clim = random_fieldlist(101, 5)
    clim.metadata()[0]._d.update(
        {
            "powerOfTenUsedToScaleClimateWeight": 0,
            "weightAppliedToClimateMonth1": 0,
            "firstMonthUsedToBuildClimateMonth1": 0,
            "lastMonthUsedToBuildClimateMonth1": 0,
            "firstMonthUsedToBuildClimateMonth2": 0,
            "lastMonthUsedToBuildClimateMonth2": 0,
            "numberOfBitsContainingEachPackedValue": 0,
        }
    )
    NumpyFieldListBackend.efi(clim, ens, 0.0001)
    NumpyFieldListBackend.sot(clim, ens, 90, 0.0001)


def test_quantiles():
    ens = random_fieldlist(5, 5)
    NumpyFieldListBackend.quantiles(ens, 0.1)


def test_filter():
    arr = [random_fieldlist(1, 5) for _ in range(2)]
    NumpyFieldListBackend.filter(arr[0], arr[1], "<", 2, replacement=0)


def test_concat():
    arr = [random_fieldlist(1, 5) for _ in range(3)]
    res = NumpyFieldListBackend.concat(*arr)
    assert len(res) == 3
    assert np.all(res[0].values == arr[0].values)
    assert np.all(res[-1].values == arr[-1].values)


@pytest.mark.parametrize(
    ["args", "kwargs", "output_shape"],
    [
        [[0], {"axis": 0}, (1, 2, 3)],
        [[[0]], {"axis": 0}, (1, 2, 3)],
        [[[0, 1]], {"axis": 0}, (2, 2, 3)],
    ],
)
def test_take(args, kwargs, output_shape):
    input = random_fieldlist(3, 2, 3)
    output = NumpyFieldListBackend.take(input, *args, **kwargs)
    assert isinstance(output, FieldList)
    assert output.values.shape == output_shape
