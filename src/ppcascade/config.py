import bisect
from datetime import timedelta
import copy

from pproc.common.config import Config as BaseConfig
from pproc.clustereps.config import FullClusterConfig

from .utils.io import _source_from_location
from .utils.request import Request, MultiSourceRequest
from .utils.window import WindowConfig


class ParamConfig:
    def __init__(self, members, param_config, in_keys, out_keys):
        param_options = copy.deepcopy(param_config)
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
        self.out_keys = copy.deepcopy(out_keys)
        self.out_keys.update(param_options.pop("grib_set", {}))
        self.in_keys = copy.deepcopy(in_keys)
        self.options = param_options

    @classmethod
    def _generate_steps(cls, steps_config):
        unique_steps = set()
        for steps in steps_config:
            if len(steps) == 3:
                for step in range(steps[0], steps[1] + 1, steps[2]):
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
        window = WindowConfig(operation, include_init, windows_config)
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


class WindParamConfig(ParamConfig):
    def vod2uv(self, fc: str) -> bool:
        _, reqs = _source_from_location(fc, self.sources)
        return reqs[0].get("interpolate", {}).get("vod2uv", "0") == "1"

    def forecast_request(self, fc: str):
        vod2uv = self.vod2uv(fc)
        no_expand = ("param") if vod2uv else ()
        self.param["kwargs"].update({"vod2uv": vod2uv})
        return super().forecast_request(fc, no_expand), not vod2uv


class ExtremeParamConfig(ParamConfig):
    def clim_request(
        self, clim: str, accumulated: bool = False, no_expand: tuple[str] = ()
    ):
        clim_reqs = super().clim_request(clim, accumulated, no_expand)
        for req in clim_reqs:
            num_quantiles = int(req["quantile"])
            req["quantile"] = ["{}:100".format(i) for i in range(num_quantiles + 1)]
        return clim_reqs


def new_param_config(members, param_config, in_keys, out_keys):
    if param_config.get("param", {}).get("operation", None) == "wind_speed":
        return WindParamConfig(members, param_config, in_keys, out_keys)
    if param_config.get("ensemble", {}).get("operation", None) in [
        "efi",
        "sot",
        "extreme",
    ]:
        return ExtremeParamConfig(members, param_config, in_keys, out_keys)
    return ParamConfig(members, param_config, in_keys, out_keys)


class Config(BaseConfig):
    def __init__(self, args):
        super().__init__(args)

        if isinstance(self.options["members"], dict):
            members = range(
                self.options["members"]["start"], self.options["members"]["end"] + 1
            )
        else:
            members = range(1, int(self.options["members"]) + 1)
        out_keys = self.options.pop("out_keys", {})
        in_keys = self.options.pop("in_keys", {})
        self.parameters = [
            new_param_config(members, cfg, in_keys, out_keys)
            for cfg in self.options.pop("parameters", [])
        ]


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
