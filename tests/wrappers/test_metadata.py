import datetime
import dill

from ppcascade.backends.fieldlist import ArrayFieldListBackend


def test_serialisation(tmpdir):
    yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
    data = ArrayFieldListBackend.retrieve(
        {
            "class": "od",
            "date": yesterday.strftime("%Y%m%d"),
            "domain": "g",
            "expver": "0001",
            "levtype": "sfc",
            "param": "167",
            "stream": "enfo",
            "time": "12",
            "type": "cf",
            "source": "mars",
            "step": 36,
        },
        backend_kwargs={"stream": True},
    )
    metadata = data[0].metadata()

    dill.dump(metadata, open(tmpdir / "metadata.pkl", "wb"))
    deserialized = dill.load(open(tmpdir / "metadata.pkl", "rb"))
    assert metadata._handle.get_buffer() == deserialized._handle.get_buffer()
