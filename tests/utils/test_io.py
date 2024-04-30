from datetime import datetime, timedelta
import numpy as np
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
            "interpolate": {"grid": "O1280", "vod2uv": "1"},
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


@pytest.mark.skip(reason="Hangs if run in test suite")
def test_multiprocess(tmpdir):
    import multiprocessing
    import dill

    dill.Pickler.dumps, dill.Pickler.loads = dill.dumps, dill.loads
    multiprocessing.reduction.ForkingPickler = dill.Pickler
    multiprocessing.reduction.dump = dill.dump
    multiprocessing.queues._ForkingPickler = dill.Pickler
    import concurrent.futures as fut

    futures = []
    base_request = request.copy()
    with fut.ProcessPoolExecutor(max_workers=2) as executor:
        for x in range(1, 3):
            base_request["type"] = "pf"
            base_request["number"] = x
            futures.append(executor.submit(retrieve, base_request))

    for future in fut.as_completed(futures):
        data = future.result()
