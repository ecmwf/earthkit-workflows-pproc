from typing import Any
import yaml
import copy


from .parsemars import ComputeRequest, ProductType, MarsKey


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
        additional_updates: dict | None = None,
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
        if additional_updates is not None:
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


class ExtremeConfig(ProductConfig):
    def add_param(
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
        super().add_param(common, param, request, updates)


class QuantileConfig(ProductConfig):
    def add_param(
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
            start, end = request.base_request[MarsKey.NUMBER]
            request.base_request[MarsKey.NUMBER] = list(range(start, end + 1))
        return super().add_param(common, param, request)


def deep_update(original: dict, update: Any) -> dict:
    for key, value in update.items():
        if isinstance(value, dict):
            original[key] = deep_update(original.get(key, {}), value)
        else:
            original[key] = value
    return original


class RequestTranslator:
    def __init__(self, filename: str):
        self.config_defaults = ConfigDefaults(filename)

    def _new_product(self, prod_type: ProductType):
        config_type = ProductConfig
        if prod_type in [
            ProductType.EFI,
            ProductType.EFI_CONTROL,
            ProductType.SOT,
        ]:
            product, config_type = "extreme", ExtremeConfig
        elif prod_type == ProductType.EVENT_PROB:
            product = "prob"
        elif prod_type == ProductType.QUANTILES:
            product, config_type = "quantiles", QuantileConfig
        elif prod_type == [ProductType.ENS_MEAN, ProductType.ENS_STD]:
            product, config_type = "ensms", EnsmsConfig
        else:
            product = "forecast", ForecastConfig
        prod_config = copy.deepcopy(self.config_defaults.base["global"])
        prod_config.update(
            self.config_defaults.products.get(product, {}).get("global", {})
        )
        return product, config_type(**prod_config)

    def translate(self, requests: list[ComputeRequest]):
        config = {}
        for request in requests:
            product, prod_config = self._new_product(request.type)

            config.setdefault(product, prod_config)
            common = copy.deepcopy(self.config_defaults.base["common"])
            deep_update(
                common, self.config_defaults.products.get(product, {}).get("common", {})
            )

            # Try to see if param details is specified in the base params, if not try product params
            # otherwise parameter operation is trivial
            param = self.config_defaults.base["params"].get(
                int(request.param),
                self.config_defaults.products.get(product, {})
                .get("params", {})
                .get(request.param, {"param": {"id": request.param}}),
            )
            config[product].add_param(common, param, request)
        return config
