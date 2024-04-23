from typing import Any
import yaml
import copy
import bisect


from datetime import datetime, timedelta

from .parsemars import ComputeRequest, ProductType, MarsKey, parse_request


class ConfigDefaults:
    def __init__(self, filename: str):
        with open(filename, "r") as f:
            config = yaml.safe_load(f)
        self.products = config.pop("products", {})
        self.type_to_product = {}
        for mars_types in self.products.keys():
            for type in mars_types.split("/"):
                self.type_to_product[type] = mars_types
        self.base = config

    def product(self, request_type: str) -> dict:
        prod_type = self.type_to_product.get(request_type, None)
        if prod_type is None:
            assert request_type in [
                ProductType.CTRL_FORECAST,
                ProductType.DET_FORECAST,
                ProductType.PERTURBED_FORECAST,
            ]
            return {}
        return self.products[prod_type]


class ProductConfig:
    graph_product = "ensemble"

    def __init__(self, members: str | dict, in_keys: dict = {}, out_keys: dict = {}):
        self._members = members
        self.in_keys = in_keys
        self.out_keys = out_keys
        self.parameters = []

    @property
    def members(self):
        if isinstance(self._members, dict):
            return range(self._members["start"], self._members["end"] + 1)
        return range(1, int(self._members) + 1)

    def num_members(self):
        mem = list(self.members)
        return mem[-1] - mem[0] + 1

    def add_param(
        self,
        common: dict,
        param: dict,
        request: ComputeRequest,
        additional_updates: dict | None = None,
    ):
        config = copy.deepcopy(common)
        config["target"] = request.target
        deep_update(config, param)
        base_interp = config.pop("interpolate")
        param_id = config["param"].pop("id")
        if len(config["param"]) == 0:
            config.pop("param")
        config.setdefault("windows", {})
        config.setdefault("grib_set", {})

        if config["windows"].get("operation", None) == "diff":
            windows = [x.split("-") for x in request.steps]
            config["windows"]["ranges"] = [
                [int(x[0]), int(x[1]), int(x[1]) - int(x[0])] for x in windows
            ]
        else:
            config["windows"]["ranges"] = [
                [x, x] if isinstance(x, int) else list(map(int, x.split("-")))
                for x in request.steps
            ]
        if request.grid is not None or request.base_request[MarsKey.LEVTYPE] == "pl":
            interpolation_keys = copy.deepcopy(base_interp)
            if request.grid is not None:
                interpolation_keys["grid"] = request.grid
            request.base_request["interpolate"] = interpolation_keys
        config["sources"] = self._create_sources(
            param_id, config.pop("sources", {}), request.base_request
        )

        if "ensemble" in config and isinstance(config["ensemble"]["operation"], dict):
            config["ensemble"]["operation"] = config["ensemble"]["operation"][
                request.type
            ]
        if additional_updates is not None:
            deep_update(config, additional_updates)
        self.parameters.append(config)

    def _create_sources(self, param_id: str, sources: dict, base_request: dict):
        params = (
            list(map(str, param_id)) if isinstance(param_id, list) else str(param_id)
        )
        for src, values in sources.items():
            src_keys = {"domain": "g"} if src == "fdb" else {}
            for param_type, param_keys in values.items():
                if isinstance(param_keys, dict):
                    sources[src][param_type] = {
                        **base_request,
                        "param": params,
                        **param_keys,
                        **src_keys,
                    }
                else:
                    sources[src][param_type] = [
                        {**base_request, "param": params, **x, **src_keys}
                        for x in sources[src][param_type]
                    ]
        return sources

    def to_yaml(self, filename: str):
        ret = {
            "members": self._members,
        }
        ret.update(
            {
                k: getattr(self, k)
                for k in [
                    "in_keys",
                    "out_keys",
                    "parameters",
                ]
            }
        )
        yaml.dump(ret, open(filename, "w"))


class ExtremeConfig(ProductConfig):
    graph_product = "extreme"

    def add_param(
        self,
        common: dict,
        param: dict,
        request: ComputeRequest,
        additional_updates: dict | None = None,
    ):
        updates = (
            copy.deepcopy(additional_updates) if additional_updates is not None else {}
        )
        if request.type == "sot":
            deep_update(
                updates, {"ensemble": {"sot": request.base_request.pop(MarsKey.NUMBER)}}
            )

        if request.type == "efic":
            common = copy.deepcopy(common)
            for src in common["sources"].keys():
                common["sources"][src]["ens"] = {**(common["sources"][src]["ens"][1])}

        for src in common["sources"].keys():
            deep_update(
                param,
                {
                    "sources": {
                        src: {
                            "clim": {
                                "date": ExtremeConfig.clim_date(
                                    request.base_request[MarsKey.DATE]
                                ),
                                "time": "00",
                                "step": {
                                    x: ExtremeConfig.clim_step(
                                        x, request.base_request[MarsKey.TIME]
                                    )
                                    for x in request.steps
                                },
                            }
                        }
                    }
                },
            )
        super().add_param(common, param, request, updates)

    @staticmethod
    def clim_date(date_str: str):
        date = datetime.strptime(date_str, "%Y%m%d")
        weekday = date.weekday()
        # friday to monday -> take previous monday clim, else previous thursday clim
        if weekday == 0 or weekday > 3:
            clim_date = date - timedelta(days=(weekday + 4) % 7)
        else:
            clim_date = date - timedelta(days=weekday)
        return clim_date.strftime("%Y%m%d")

    @staticmethod
    def clim_window_starts(extended: bool = False):
        if extended:
            return [0, 96, 168, 264, 336, 432, 504, 600, 672, 768, 840, 936]
        return list(range(0, 217, 12))

    @staticmethod
    def clim_step(interval, time: str, extended: bool = False):
        start, end = map(int, interval.split("-"))
        clim_relative_time = start + int(time)
        if time == "12":
            clim_relative_time = start - int(time)
        if end < 240 or extended:
            clim_windows = ExtremeConfig.clim_window_starts(extended)
            nearest_clim_window = bisect.bisect_right(clim_windows, clim_relative_time)
            clim_window_start = clim_windows[nearest_clim_window - 1]
            if (nearest_clim_window <= (len(clim_windows) - 1)) and (
                (clim_windows[nearest_clim_window] - clim_relative_time)
                < (clim_relative_time - clim_window_start)
            ):
                clim_window_start = clim_windows[nearest_clim_window]
            return f"{clim_window_start}-{clim_window_start + (end - start)}"
        return f"{start}-{end}"


class QuantileConfig(ProductConfig):
    def add_param(
        self,
        common: dict,
        param: dict,
        request: ComputeRequest,
        additional_updates: dict | None = None,
    ):
        updates = (
            copy.deepcopy(additional_updates) if additional_updates is not None else {}
        )
        quantiles = int(request.base_request.pop("quantile"))
        deep_update(
            updates,
            {
                "ensemble": {"num_quantiles": quantiles},
                "grib_set": {"numberOfForecastsInEnsemble": quantiles},
            },
        )
        super().add_param(common, param, request, updates)


class EnsmsConfig(ProductConfig):
    def add_param(
        self,
        common: dict,
        param: dict,
        request: ComputeRequest,
        additional_updates: dict | None = None,
    ):
        updates = (
            copy.deepcopy(additional_updates) if additional_updates is not None else {}
        )
        deep_update(
            updates, {"grib_set": {"numberOfForecastsInEnsemble": self.num_members()}}
        )
        return super().add_param(common, param, request, updates)


class ForecastConfig(ProductConfig):
    def add_param(
        self,
        common: dict,
        param: dict,
        request: ComputeRequest,
        additional_updates: dict | None = None,
    ):
        request.base_request[MarsKey.TYPE] = request.type
        if request.type == ProductType.PERTURBED_FORECAST:
            request.base_request[MarsKey.NUMBER] = request.base_request[MarsKey.NUMBER]
        return super().add_param(common, param, request, additional_updates)


class EnsembleAnomalyConfig(ProductConfig):
    graph_product = "ensemble_anomaly"

    def add_param(
        self,
        common: dict,
        param: dict,
        request: ComputeRequest,
        additional_updates: dict | None = None,
    ):
        updates = (
            copy.deepcopy(additional_updates) if additional_updates is not None else {}
        )
        steps = []
        for step_range in request.steps:
            if isinstance(step_range, int):
                steps.append(step_range)
            else:
                steps.extend(
                    range(int(step_range[0]), int(step_range[1]) + 1, 12)
                )  # TODO: 12 is hardcoded

        for src in common["sources"].keys():
            deep_update(
                param,
                {
                    "sources": {
                        src: {
                            "clim": {
                                "date": EnsembleAnomalyConfig.clim_date(
                                    request.base_request[MarsKey.DATE]
                                ),
                                "time": "00",
                                "step": {
                                    x: EnsembleAnomalyConfig.clim_step(
                                        x, request.base_request[MarsKey.TIME]
                                    )
                                    for x in steps
                                },
                            }
                        }
                    }
                },
            )
        super().add_param(common, param, request, updates)

    @staticmethod
    def clim_date(date_str: str):
        """
        Assumes climatology run on Monday and Thursday and retrieves most recent
        date climatology available
        """
        date = datetime.strptime(date_str, "%Y%m%d")
        dow = date.weekday()
        if dow >= 0 and dow < 3:
            return (date - timedelta(days=dow)).strftime("%Y%m%d")
        return (date - timedelta(days=(dow - 3))).strftime("%Y%m%d")

    @staticmethod
    def clim_step(step: int, time: str):
        """
        Nearest step with climatology data to step,
        taking into account diurnal variation in climatology
        which requires climatology step time to be same
        as step
        """
        if time in ["12", "18"]:
            if step == 360:
                return step - 12
            return step + 12
        return step


def deep_update(original: dict, update: Any) -> dict:
    for key, value in update.items():
        if isinstance(value, dict):
            original[key] = deep_update(original.get(key, {}), value)
        else:
            original[key] = value
    return original


class RequestTranslator:
    config_mapping = {
        ProductType.EFI: ExtremeConfig,
        ProductType.EFI_CONTROL: ExtremeConfig,
        ProductType.SOT: ExtremeConfig,
        ProductType.QUANTILES: QuantileConfig,
        ProductType.ENS_MEAN: EnsmsConfig,
        ProductType.ENS_STD: EnsmsConfig,
        ProductType.EVENT_PROB: ProductConfig,
        ProductType.CTRL_FORECAST: ForecastConfig,
        ProductType.DET_FORECAST: ForecastConfig,
        ProductType.PERTURBED_FORECAST: ForecastConfig,
    }

    def __init__(self, filename: str):
        self.config_defaults = ConfigDefaults(filename)

    def _product_config(self, config: dict, request_type: str, param_config: dict):
        config_type = self.config_mapping.get(request_type)
        if request_type == ProductType.EVENT_PROB:
            sources = param_config.get("sources", {})
            if len(sources) > 0 and "clim" in list(sources.values())[0]:
                config_type = EnsembleAnomalyConfig

        prod_config = copy.deepcopy(self.config_defaults.base["global"])
        prod_config.update(self.config_defaults.product(request_type).get("global", {}))
        config.setdefault(config_type.__name__, config_type(**prod_config))
        return config[config_type.__name__]

    def translate(self, request_file: str):
        requests = parse_request(request_file)
        config = {}
        for request in requests:
            common = copy.deepcopy(self.config_defaults.base["common"])
            deep_update(
                common, self.config_defaults.product(request.type).get("common", {})
            )

            # Try to see if param details is specified in the base params, if not try product params
            # otherwise parameter operation is trivial
            param = self.config_defaults.base["params"].get(
                int(request.param),
                self.config_defaults.product(request.type)
                .get("params", {})
                .get(int(request.param), {"param": {"id": request.param}}),
            )

            prod_config = self._product_config(config, request.type, param)
            prod_config.add_param(common, param, request)
        return config
