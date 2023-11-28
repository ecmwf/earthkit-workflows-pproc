from io import BytesIO
import numpy as np
import os
import importlib
from os.path import join as pjoin
import shutil
import sys

import mir
from pproc.common.io import (
    target_from_location,
    write_grib,
    FileTarget,
    FileSetTarget,
)
from pproc.common import ResourceMeter
from pproc.clustereps.__main__ import write_cluster_attr_grib
from pproc.clustereps.cluster import get_output_keys
from pproc.common.io import split_location
from earthkit.data import FieldList
from earthkit.data.sources.stream import StreamSource
from earthkit.data.sources.file import FileSource
from earthkit.data.sources.fdb import FDBSource
from earthkit.data.sources.mars import MarsRetriever
from earthkit.data.sources import Source, from_source
from earthkit.data.sources.numpy_list import NumpyFieldList

from .grib import basic_headers
from .wrappers.metadata import GribBufferMetaData


def mir_job(
    input: mir.MultiDimensionalGribFileInput, mir_options: dict, cache: str = None
) -> Source:
    job = mir.Job(**mir_options)
    stream = BytesIO()
    job.execute(input, stream)
    stream.seek(0)
    if cache is None:
        return StreamSource(stream, batch_size=0)

    with open(cache, "wb") as o, stream as i:
        shutil.copyfileobj(i, o)
    return FileSource(cache)


class MarsWithMir(MarsRetriever):
    def mutate(self):
        mir_options = request.pop("interpolate", None)
        cache = request.pop("cache", None)
        cache_path = None if cache is None else cache.format_map(request)
        ds = super()._retrieve(request)
        if mir_options:
            size = len(request["param"]) if isinstance(request["param"], list) else 1
            inp = mir.MultiDimensionalGribFileInput(ds.path, size)
            ds = mir_job(inp, mir_options, cache_path)
        return ds


class FDBWithMir(FDBSource):
    def mutate(self):
        mir_options = self.request.pop("interpolate", None)
        if mir_options is None:
            return from_source("fdb", self.request, batch_size=0, stream=self.stream)

        reader = from_source("fdb", self.request, stream=self.stream)
        if self.stream:
            ds = mir_job(reader._stream, mir_options)
        else:
            size = (
                len(self.request["param"])
                if isinstance(self.request["param"], list)
                else 1
            )
            inp = mir.MultiDimensionalGribFileInput(reader.path, size)
            ds = mir_job(inp, mir_options)
        return ds


class FileWithMir(File):
    def __init__(
        self,
        path,
        request: dict,
        expand_user=True,
        expand_vars=False,
        unix_glob=True,
        recursive_glob=True,
        filter=None,
        merger=None,
        **kwargs,
    ):
        request_path = path.format_map(request)
        super().__init__(
            request_path,
            expand_user,
            expand_vars,
            unix_glob,
            recursive_glob,
            filter,
            merger,
            **kwargs,
        )
        self.request = request

    def _transform_steps(cls, steps, step_type: type = str):
        if isinstance(steps, (int, str)):
            steps = [steps]
        return list(map(step_type, steps))

    def _transform_request(cls, request: dict, step_type: type = str):
        try:
            paramId = int(request["param"])
            del request["param"]
            request["paramId"] = paramId
        except:
            pass
        request["date"] = int(request["date"])
        time = int(request["time"])
        request["time"] = time if time % 100 == 0 else time * 100
        request["step"] = cls._transform_steps(request["step"], step_type)
        return request

    def mutate(self):
        mir_options = self.request.pop("interpolate", None)
        if mir_options:
            raise NotImplementedError()
        file_ds = super().mutate()
        ds = file_ds.sel(self._transform_request(self.request))
        if len(ds) == 0:
            try:
                self.request["step"] = self._transform_steps(self.request["step"], int)
                ds = file_ds.sel(self.request)
            except ValueError:
                pass
        return ds
