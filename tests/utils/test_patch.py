# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest
import xarray as xr
from earthkit.meteo.extreme import array as extreme

from earthkit.workflows.plugins.pproc.utils.patch import PatchModule


def test_patch_extreme():
    a = xr.DataArray(np.arange(6).reshape(2, 3), dims=["x", "y"])
    with pytest.raises(AttributeError) as err:
        with PatchModule(extreme, "numpy", xr):
            extreme.efi(a, a, 0.0001)

    assert str(err.value) == "module 'xarray' has no attribute 'logical_or'"
