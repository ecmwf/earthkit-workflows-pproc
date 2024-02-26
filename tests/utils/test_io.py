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


def test_retrieve():
    # Retrieve from single source
    data = retrieve(request)

    # Retrieve with multiple sources
    fdb_request = request.copy()
    fdb_request["source"] = "fdb"
    data2 = retrieve([request, fdb_request])
    assert np.all(data.values == data2.values)


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
        write(f"{tmpdir}/test.grib", data, {"step": 12})
