from datetime import datetime, timedelta

import pytest

from ppcascade.utils.io import retrieve

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
