# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from datetime import datetime, timedelta

import pytest

from earthkit.workflows.plugins.pproc.utils.io import retrieve

request = {
    "class": "od",
    "expver": "0001",
    "stream": "enfo",
    "type": "cf",
    "date": (datetime.today() - timedelta(days=1)).strftime("%Y%m%d"),
    "time": "12",
    "domain": "g",
    "levtype": "sfc",
    "step": "12",
    "param": 228,
    "source": "mars",
}


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"interpolate": {"grid": "O640"}},
        {
            "param": [138, 155],
            "levtype": "pl",
            "levelist": [250, 850],
            "interpolate": {"grid": "O640", "vod2uv": "1"},
        },
    ],
    ids=["default", "interpolate", "wind"],
)
def test_retrieve(overrides):
    test_request = request.copy()
    test_request.update(overrides)
    retrieve(test_request)


def test_retrieve_multi():
    fdb_request = request.copy()
    fdb_request["source"] = "fdb"
    retrieve([request, fdb_request])
