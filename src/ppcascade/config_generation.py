from typing import Any
import yaml
import copy

from .parse_requests import ComputeRequest, ProductType, MarsKey


class ConfigDefaults:
    def __init__(self, filename: str):
        with open(filename, "r") as f:
            config = yaml.safe_load(f)
        self.products = config.pop("products", {})
        self.base = config


class ProductConfig:
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
        additional_updates: dict,
    ):
        config = {**common, "target": request.target}
        deep_update(config, param)
        base_interp = config.pop("interpolate")
        param_id = config["param"].pop("id")
        if len(config["param"]) == 0:
            config.pop("param")
        config.setdefault("windows", {})
        config.setdefault("grib_set", {})

        config["windows"]["ranges"] = request.steps
        if request.grid is not None:
            request.base_request["interpolate"] = {"grid": request.grid, **base_interp}
        config["sources"] = self._create_sources(
            param_id, config.pop("sources", {}), request.base_request
        )
        if "ensemble" in config and isinstance(config["ensemble"]["operation"], dict):
            config["ensemble"]["operation"] = config["ensemble"]["operation"][
                request.type
            ]
        deep_update(config, additional_updates)
        self.parameters.append(config)

    def _create_sources(self, param_id: str, sources: dict, base_request: dict):
        for src, values in sources.items():
            for param_type, param_keys in values.items():
                if isinstance(param_keys, dict):
                    sources[src][param_type] = {
                        **base_request,
                        "param": param_id,
                        **param_keys,
                    }
                else:
                    sources[src][param_type] = [
                        {
                            **base_request,
                            "param": param_id,
                            **x,
                        }
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


def deep_update(original: dict, update: Any) -> dict:
    for key, value in update.items():
        if isinstance(value, dict):
            original[key] = deep_update(original.get(key, {}), value)
        else:
            original[key] = value
    return original


class Config:
    def __init__(self, filename: str):
        self.config_defaults = ConfigDefaults(filename)

    def add_product(self, config: dict, product: str):
        if product not in config:
            prod_config = copy.deepcopy(self.config_defaults.base["global"])
            prod_config.update(self.config_defaults.products[product]["global"])
            config[product] = ProductConfig(**prod_config)

    def add_extreme_param(self, config, request: dict):
        common = copy.deepcopy(self.config_defaults.base["common"])
        deep_update(common, self.config_defaults.products["extreme"]["common"])
        param = self.config_defaults.products["extreme"]["params"][request.param]
        additional_updates = {}
        if request.type == "sot":
            additional_updates = {
                "ensemble": {"sot": request.base_request.pop(MarsKey.NUMBER)}
            }
        config.add_param(common, param, request, additional_updates)

    def add_prob_param(self, config, request: dict):
        common = copy.deepcopy(self.config_defaults.base["common"])
        deep_update(common, self.config_defaults.products["prob"]["common"])
        param = self.config_defaults.products["prob"]["params"][request.param]
        config.add_param(common, param, request, {})

    def add_quantile_param(self, config, request: dict):
        quantiles = int(request.base_request.pop("quantile"))
        common = copy.deepcopy(self.config_defaults.base["common"])
        deep_update(common, self.config_defaults.products["quantiles"]["common"])
        param = self.config_defaults.base["params"].get(
            int(request.param), {"param": {"id": request.param}}
        )
        config.add_param(
            common,
            param,
            request,
            {
                "ensemble": {"num_quantiles": quantiles},
                "grib_set": {"numberOfForecastsInEnsemble": quantiles},
            },
        )

    def add_ensms_param(self, config, request):
        common = copy.deepcopy(self.config_defaults.base["common"])
        deep_update(common, self.config_defaults.products["ensms"]["common"])
        param = self.config_defaults.base["params"].get(
            int(request.param), {"param": {"id": request.param}}
        )
        config.add_param(
            common,
            param,
            request,
            {"grib_set": {"numberOfForecastsInEnsemble": config.num_members()}},
        )

    def add_simple_param(self, config, request):
        assert request.type in [
            ProductType.DET_FORECAST,
            ProductType.PERTURBED_FORECAST,
            ProductType.CTRL_FORECAST,
        ]
        common = copy.deepcopy(self.config_defaults.base["common"])
        param = self.config_defaults.base["params"].get(
            int(request.param), {"param": {"id": request.param}}
        )
        new_request = copy.deepcopy(request)
        new_request.base_request[MarsKey.TYPE] = request.type
        if new_request.type == ProductType.PERTURBED_FORECAST:
            start, end = new_request.base_request[MarsKey.NUMBER]
            new_request.base_request[MarsKey.NUMBER] = list(range(start, end + 1))
        config.add_param(common, param, new_request, {})

    def create_config(self, requests):
        config = {}
        for request in requests:
            if request.type in [
                ProductType.EFI,
                ProductType.EFI_CONTROL,
                ProductType.SOT,
            ]:
                self.add_product(config, "extreme")
                self.add_extreme_param(config["extreme"], request)
            elif request.type == ProductType.EVENT_PROB:
                self.add_product(config, "prob")
                self.add_prob_param(config["prob"], request)
            elif request.type == ProductType.QUANTILES:
                self.add_product(config, "quantiles")
                self.add_quantile_param(config["quantiles"], request)
            elif request.type in [ProductType.ENS_MEAN, ProductType.ENS_STD]:
                self.add_product(config, "ensms")
                self.add_ensms_param(config["ensms"], request)
            else:
                self.add_product(config, "ensms")
                self.add_simple_param(config["ensms"], request)
        return config
