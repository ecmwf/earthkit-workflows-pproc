import itertools
from collections import OrderedDict
import bisect
from datetime import timedelta
import copy
import numpy as np
from dataclasses import dataclass, field

from pproc.common.config import Config as BaseConfig
from pproc.clustereps.config import FullClusterConfig

from .io import _source_from_location


@dataclass
class Range:
    name: str
    steps: list[int]


@dataclass
class WindowConfig:
    operation: str
    include_init: bool
    options: dict
    grib_sets: dict
    ranges: list[Range] = field(default_factory=list)

    # TOOD: precomputed windows, how is their EFI computed?? Do we need num steps?
    def add_range(self, start: int, end: int, step: int = 1, allowed_steps: list = []):
        window_size = end - start
        if window_size == 0:
            name = str(end)
        else:
            name = f"{start}-{end}"

        # Set steps in window
        if len(allowed_steps) == 0:
            allowed_steps = list(range(start, end + 1, step))
        if self.include_init or (window_size == 0):
            start_index = allowed_steps.index(start)
        else:
            # Case when window.start not in steps
            start_index = bisect.bisect_right(allowed_steps, start)
        steps = allowed_steps[start_index : allowed_steps.index(end) + 1]
        assert name not in self.ranges
        self.ranges.append(Range(name, steps))


class Request:
    def __init__(self, request: dict, no_expand: tuple[str] = ()):
        self.request = request.copy()
        self.fake_dims = []
        self.no_expand = no_expand
        self.ignore = ["interpolate"]

    @property
    def dims(self) -> OrderedDict:
        dimensions = OrderedDict()
        for key, values in self.request.items():
            if key in self.ignore or key in self.no_expand:
                continue
            if hasattr(values, "__iter__") and not isinstance(values, str):
                dimensions[key] = len(values)
        return dimensions

    def __setitem__(self, key, value):
        self.request[key] = value

    def __getitem__(self, key):
        return self.request[key]

    def __contains__(self, key) -> bool:
        return key in self.request

    def update(self, **kwargs):
        self.request.update(**kwargs)

    def pop(self, key, default=None):
        if default is None:
            return self.request.pop(key)
        return self.request.pop(key, default)

    def make_dim(self, key, value=None):
        if key in self:
            assert type(self[key], (str, int, float))
            self[key] = [self[key]]
        else:
            self[key] = [value]
            self.fake_dims.append(key)

    def expand(self):
        for params in itertools.product(*[self.request[x] for x in self.dims.keys()]):
            new_request = self.request.copy()
            indices = []
            for index, expand_param in enumerate(self.dims.keys()):
                new_request[expand_param] = params[index]
                indices.append(list(self.request[expand_param]).index(params[index]))

            # Remove fake dims from request
            for dim in self.fake_dims:
                new_request.pop(dim)
            yield tuple(indices), new_request


class MultiSourceRequest(Request):
    def __init__(self, requests: list[dict], no_expand: tuple[str] = ()):
        super().__init__(requests[0], no_expand)
        self.requests = requests

    def __setitem__(self, key, value):
        super().__setattr__(key, value)
        [x.__setitem__(key, value) for x in self.requests]

    def __getitem__(self, key):
        values = [x.__getitem__(key) for x in self.requests]
        if np.all([values[0] == values[x] for x in range(1, len(values))]):
            return values[0]
        raise Exception(f"Requests {self.requests} differ on value for key {key}")

    def __contains__(self, key) -> bool:
        contains = [x.__contains__(key) for x in self.requests]
        if all([contains[0] == contains[x] for x in range(1, len(contains))]):
            return contains[0]
        raise Exception(f"Not all requests {self.requests} contain key {key}")

    def update(self, **kwargs):
        super().update(**kwargs)
        [x.update(**kwargs) for x in self.requests]

    def pop(self, key, default=None):
        contains = key in self
        if default is None or contains:
            value = self[key]
            super().pop(key)
            [x.pop(key) for x in self.requests]
            return value
        super().pop(key)
        [x.pop(key) for x in self.requests]
        return default

    def expand(self):
        for params in itertools.product(*[self.request[x] for x in self.dims.keys()]):
            indices = []
            new_requests = copy.deepcopy(self.requests)
            for index, expand_param in enumerate(self.dims.keys()):
                [x.__setitem__(expand_param, params[index]) for x in new_requests]
                indices.append(list(self.request[expand_param]).index(params[index]))

            # Remove fake dims from request
            for dim in self.fake_dims:
                [x.pop(dim) for x in new_requests]
            yield tuple(indices), new_requests


class Config(BaseConfig):
    def __init__(self, args):
        super().__init__(args)

        if isinstance(self.options["members"], dict):
            self.members = range(
                self.options["members"]["start"], self.options["members"]["end"] + 1
            )
        else:
            self.members = range(1, int(self.options["members"]) + 1)
        self.out_keys = self.options.pop("out_keys", {})
        self.in_keys = self.options.pop("in_keys", {})


class ParamConfig:
    def __init__(self, members, param_config, in_keys, out_keys):
        param_options = param_config.copy()
        self.steps = self._generate_steps(param_options.pop("steps", []))
        self.sources = param_options.pop("sources")
        self.members = members
        self.windows = self._generate_windows(param_options.pop("windows"))
        self.param = {
            "operation": param_options.get("param", {}).pop("operation", None),
            "kwargs": param_options.get("param", {}),
        }
        self.ensemble = {
            "operation": param_options.get("ensemble", {}).pop("operation", None),
            "kwargs": param_options.get("ensemble", {}),
        }
        self.target = param_options.pop("target")
        self.out_keys = out_keys.copy()
        self.in_keys = in_keys.copy()
        self.options = param_options

    @classmethod
    def _generate_steps(cls, steps_config):
        unique_steps = set()
        for steps in steps_config:
            start_step = steps["start_step"]
            end_step = steps["end_step"]
            interval = steps["interval"]
            range_len = steps.get("range", None)

            if range_len is None:
                for step in range(start_step, end_step + 1, interval):
                    if step not in unique_steps:
                        unique_steps.add(step)
            else:
                raise NotImplementedError
                # for sstep in range(start_step, end_step - range_len + 1, interval):
                #     steps.add(Step(sstep, sstep + range_len))
        return sorted(unique_steps)

    def _generate_windows(self, windows_config: dict):
        include_init = windows_config.pop("include_start_step", False)
        operation = windows_config.pop("operation", None)
        ranges = windows_config.pop("ranges")
        grib_sets = windows_config.pop("grib_set", {})
        window = WindowConfig(operation, include_init, windows_config, grib_sets)
        for r in ranges:
            window.add_range(*map(int, r), allowed_steps=self.steps)
        return window

    def _request_steps(self, window):
        if len(self.steps) == 0:
            return window.steps
        # Case when window.start not in steps
        if window.include_init:
            start_index = self.steps.index(window.start)
        else:
            start_index = bisect.bisect_right(self.steps, window.start)
        return self.steps[start_index : self.steps.index(window.end) + 1]

    def forecast_request(self, ens: str, no_expand: tuple[str] = ()):
        source, requests = _source_from_location(ens, self.sources)
        if isinstance(requests, dict):
            requests = [requests]

        window_requests = []
        for request in requests:
            req = Request({**request, **self.in_keys, "source": source}, no_expand)
            window_steps = [x.steps for x in self.windows.ranges]
            req["step"] = sorted(set(sum(window_steps[1:], window_steps[0])))
            if request["type"] == "pf":
                req["number"] = self.members
            elif request["type"] == "cf":
                req.make_dim("number", 0)
            window_requests.append(req)
        return window_requests

    def clim_request(
        self, clim: str, accumulated: bool = False, no_expand: tuple[str] = ()
    ):
        source, requests = _source_from_location(clim, self.sources)
        assert len(requests) == 1, f"Expected a single request, got {requests}"
        clim_req = Request(requests[0], no_expand)
        clim_req["source"] = source
        steps = clim_req.pop("step", {})
        if accumulated:
            window_ranges = [w.name for w in self.windows.ranges]
            clim_req["step"] = list(set(map(steps.get, window_ranges, window_ranges)))
        else:
            window_steps = [
                list(map(steps.get, w.steps, w.steps)) for w in self.windows.ranges
            ]
            clim_req["step"] = sorted(set(sum(window_steps[1:], window_steps[0])))
        if len(clim_req["step"]) == 1:
            clim_req["step"] = clim_req["step"][0]
        return [clim_req]


class WindConfig(ParamConfig):
    def vod2uv(self, fc: str) -> bool:
        _, reqs = _source_from_location(fc, self.sources)
        return reqs[0].get("interpolate", {}).get("vod2uv", "0") == "1"

    def forecast_request(self, fc: str):
        vod2uv = self.vod2uv(fc)
        no_expand = ("param") if vod2uv else ()
        self.param["kwargs"].update({"vod2uv": vod2uv})
        return super().forecast_request(fc, no_expand), not vod2uv


class ExtremeConfig(ParamConfig):
    def clim_request(
        self, clim: str, accumulated: bool = False, no_expand: tuple[str] = ()
    ):
        clim_reqs = super().clim_request(clim, accumulated, no_expand)
        for req in clim_reqs:
            num_quantiles = int(req["quantile"])
            req["quantile"] = ["{}:100".format(i) for i in range(num_quantiles + 1)]
        return clim_reqs


class ClusterConfig(FullClusterConfig):
    def spread_request(self, spread: str, no_expand: tuple[str] = ()):
        source, reqs = _source_from_location(spread, self.sources)
        assert len(reqs) == 1, f"Expected a single request, got {reqs}"
        ret = Request(reqs[0], no_expand=no_expand)
        ret.update(source=source, step=self.steps)
        return [ret]

    def spread_compute_request(
        self, spread_compute: list[str], ndays: int = 31, no_expand: tuple[str] = ()
    ):
        ret = []
        dates = [
            (self.date - timedelta(days=diff)).strftime("%Y%m%d")
            for diff in range(ndays, 0, -1)
        ]
        for loc in spread_compute:
            source, reqs = _source_from_location(loc, self.sources)
            assert len(reqs) == 1, f"Expected a single request, got {reqs}"
            reqs[0].update(step=self.steps, date=dates, source=source)
            ret.append(reqs[0])
        return [MultiSourceRequest(ret, no_expand=no_expand)]

    def forecast_request(self, ensemble: str, no_expand: tuple[str] = ()):
        source, requests = _source_from_location(ensemble, self.sources)

        window_requests = []
        for request in requests:
            req = Request({**request, "source": source}, no_expand)
            req["step"] = self.steps
            if request["type"] == "pf":
                req["number"] = list(range(1, self.num_members))
            elif request["type"] == "cf":
                req.make_dim("number", 0)
            window_requests.append(req)
        return window_requests
