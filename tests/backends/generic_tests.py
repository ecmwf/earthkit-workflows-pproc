import pytest
from earthkit.workflows import backends


@pytest.mark.parametrize(
    "comparison",
    [
        "<=",
        "<",
        ">=",
        ">",
    ],
)
def test_threshold(input_generator, comparison):
    config = {"comparison": comparison, "value": 2}
    backends.threshold(input_generator(1, 5), **config)


def test_extreme(input_generator):
    ens = input_generator(5, 5)
    clim = input_generator(101, 5)
    backends.efi(clim, ens, 0.0001)
    backends.sot(clim, ens, 90, 0.0001)


def test_quantiles(input_generator):
    ens = input_generator(5, 5)
    backends.quantiles(ens, 0.1)


def test_filter(input_generator):
    arr = [input_generator(1, 5) for _ in range(2)]
    backends.filter(arr[0], arr[1], "<", 2, replacement=0)
