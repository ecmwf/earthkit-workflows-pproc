import xarray as xr
import numpy as np
import pytest

from earthkit.meteo.extreme import array as extreme

from ppcascade.utils.patch import PatchModule


def test_patch_extreme():
    a = xr.DataArray(np.arange(6).reshape(2, 3), dims=["x", "y"])
    with pytest.raises(AttributeError) as err:
        with PatchModule(extreme, "numpy", xr):
            extreme.efi(a, a, 0.0001)

    assert str(err.value) == "module 'xarray' has no attribute 'logical_or'"
