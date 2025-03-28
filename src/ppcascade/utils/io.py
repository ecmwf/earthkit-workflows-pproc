import shutil
from io import BytesIO

import mir
from meters import ResourceMeter
from earthkit.data import FieldList, settings
from earthkit.data.sources import Source, from_source
from earthkit.data.sources.file import FileSource
from earthkit.data.sources.stream import StreamSource
from earthkit.data.readers.grib.metadata import StandAloneGribMetadata
from pproc.common.io import split_location
from pproc.common.io import FileSetTarget, FileTarget, target_from_location, write_grib

# Set cache policy to "temporary" to avoid "database is locked" errors when
# for wind when executing across multiple workers
settings.set("cache-policy", "temporary")


def _source_from_location(loc, sources) -> tuple[str, list[dict]]:
    type_, ident = split_location(loc, default="file")
    requests = sources.get(type_, {}).get(ident, None)
    assert (
        requests is not None
    ), f"Not requests listed for location {loc} in sources {sources}"
    if isinstance(requests, dict):
        requests = [requests]
    return type_, requests


def mir_job(
    input: mir.MultiDimensionalGribFileInput, mir_options: dict, cache: str = None
) -> Source:
    job = mir.Job(**mir_options)
    stream = BytesIO()
    job.execute(input, stream)
    stream.seek(0)
    if cache is None:
        return StreamSource(stream, read_all=True).mutate()

    with open(cache, "wb") as o, stream as i:
        shutil.copyfileobj(i, o)
    return FileSource(cache).mutate()


def fdb_retrieve(request: dict, *, stream: bool = True) -> Source:
    mir_options = request.pop("interpolate", None)
    if mir_options is None:
        return from_source("fdb", request, read_all=True, stream=stream)

    reader = from_source("fdb", request, stream=stream)
    if stream:
        if mir_options.get("vod2uv", "0") == "1":
            raise ValueError("Wind vod2uv not supported for stream=True")
        return mir_job(reader._source._stream, mir_options)

    if mir_options.get("vod2uv", "0") == "1":
        if len(request["param"]) != 2:
            raise ValueError("Wind vod2uv requires two parameters")
        inp = mir.MultiDimensionalGribFileInput(reader.path, 2)
    else:
        inp = mir.GribFileInput(reader.path)
    return mir_job(inp, mir_options)


def mars_retrieve(request: dict) -> Source:
    mir_options = request.pop("interpolate", None)
    cache = request.pop("cache", None)
    cache_path = None if cache is None else cache.format_map(request)
    ds = from_source("mars", request)
    if mir_options is None:
        return ds

    if mir_options.get("vod2uv", "0") == "1":
        if len(request["param"]) != 2:
            raise ValueError("Wind vod2uv requires two parameters")
        inp = mir.MultiDimensionalGribFileInput(ds.path, 2)
    else:
        inp = mir.GribFileInput(ds.path)
    return mir_job(inp, mir_options, cache_path)


def _transform_steps(steps, step_type: type = str):
    if isinstance(steps, (int, str)):
        steps = [steps]
    return list(map(step_type, steps))


def _transform_request(request: dict, step_type: type = str):
    try:
        paramId = int(request["param"])
        del request["param"]
        request["paramId"] = paramId
    except:
        pass
    if request.get("date", None) is not None:
        request["date"] = int(request["date"])
    if request.get("time", None) is not None:
        time = int(request["time"])
        request["time"] = time if time % 100 == 0 else time * 100
    if request.get("step", None) is not None:
        request["step"] = _transform_steps(request["step"], step_type)
    return request


def file_retrieve(path: str, request: dict) -> Source:
    mir_options = request.pop("interpolate", None)
    if mir_options is not None:
        raise NotImplementedError()
    location = path.format_map(request)
    file_ds = from_source("file", location)
    if len(request) > 0:
        treq = _transform_request(request)
        ds = file_ds.sel(treq)
        if len(ds) == 0:
            try:
                treq = _transform_request(request, int)
                ds = file_ds.sel(treq)
            except ValueError:
                pass
        return ds
    return file_ds


def retrieve_multi_sources(requests: list[dict], **kwargs) -> FieldList:
    ret = None
    for req in requests:
        try:
            ret = retrieve_single_source(req, **kwargs)
            break
        except AssertionError:
            continue
    assert ret is not None, f"No data retrieved from requests: {requests}"
    return ret


def retrieve_single_source(request: dict, **kwargs) -> FieldList:
    req = request.copy()
    source = req.pop("source")
    if source == "fdb":
        ret_sources = fdb_retrieve(req, **kwargs)
    elif source == "mars":
        ret_sources = mars_retrieve(req)
    elif source == "fileset":
        path = req.pop("location")
        ret_sources = file_retrieve(path, req)
    else:
        raise NotImplementedError(f"Source {source} not supported.")
    assert len(ret_sources) > 0, f"No data retrieved from {source} for request {req}"
    return ret_sources


def retrieve(request: dict | list[dict], **kwargs):
    with ResourceMeter(f"RETRIEVE {request}, {kwargs}"):
        if isinstance(request, dict):
            res = retrieve_single_source(request, **kwargs)
        else:
            res = retrieve_multi_sources(request, **kwargs)
        ret = FieldList.from_array(
            res.values,
            [StandAloneGribMetadata(metadata._handle) for metadata in res.metadata()],
        )
        return ret


def write(data: FieldList, loc, metadata: dict | None = None):
    if loc == "null:":
        return
    target = target_from_location(loc)
    if isinstance(target, (FileTarget, FileSetTarget)):
        # Allows file to be appended on each write call
        target.enable_recovery()
    assert len(data) == 1, f"Expected single field, received {len(data)}"

    template = data.metadata()[0]
    if metadata is not None:
        template = template.override(metadata)

    with ResourceMeter(f"WRITE {loc}"):
        write_grib(target, template._handle, data[0].values)
