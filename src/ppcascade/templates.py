from dataclasses import dataclass
from typing import Optional

from earthkit.workflows.fluent import Payload
from pproc.config.preprocessing import PreprocessingConfig
from pproc.schema.schema import Schema

from ppcascade.fluent import Action
from ppcascade.utils import grib


@dataclass
class StatsConfig:
    dim: str
    operation: Optional[str]
    metadata: dict


@dataclass
class EnsembleConfig:
    inputs: list[dict]
    preprocessing: PreprocessingConfig
    accumulations: dict[str, dict]
    stats: StatsConfig

    def action(
        self, forecast: Action, preprocessing_dim="param", ensemble_dim="number"
    ) -> Action:
        action = forecast
        for preprocessing in self.preprocessing:
            action = action.param_operation(dim=preprocessing_dim, **preprocessing)
        for dim, accumulation in self.accumulations.items():
            action = action.accum_operation(
                accumulation["operation"],
                dim=dim,
                coords=[accumulation["coords"]],
                metadata=accumulation.get("metadata", None),
                include_start=accumulation.get("include_start", False),
                deaccumulate=accumulation.get("deaccumulate", False),
            )
        return action.ensemble_operation(
            dim=ensemble_dim,
            **self.stats,
        )


class ExtremeConfig(EnsembleConfig):

    @property
    def climatology(self) -> list[dict]:
        return [x for x in self.inputs if x["type"] == "cd"]

    @property
    def forecast(self) -> list[dict]:
        return [x for x in self.inputs if x["type"] != "cd"]

    @property
    def step_range(self) -> str:
        step_accum = self.accumulations["step"]
        if len(step_accum["coords"]) == 1:
            return f"{step_accum['coords'][0]}"
        return f"{step_accum['coords'][0]}-{step_accum['coords'][-1]}"

    def action(
        self,
        forecast: Action,
        climatology: Action,
        preprocessing_dim="param",
        ensemble_dim="number",
    ) -> Action:
        action = forecast
        for preprocessing in self.preprocessing:
            action = action.param_operation(dim=preprocessing_dim, **preprocessing)
        for dim, accumulation in self.accumulations.items():
            action = action.accum_operation(
                accumulation["operation"],
                dim=dim,
                coords=[accumulation["coords"]],
                metadata=accumulation.get("metadata", None),
                include_start=accumulation.get("include_start", False),
                deaccumulate=accumulation.get("deaccumulate", False),
            )
        return action.ensemble_extreme(
            climatology=climatology,
            ensemble_dim=ensemble_dim,
            step_ranges=[self.step_range],
            **self.stats,
        )


class AnomalyConfig(EnsembleConfig):

    @property
    def climatology(self) -> list[dict]:
        return [x for x in self.inputs if x["type"] in ["em", "es"]]

    @property
    def forecast(self) -> list[dict]:
        return [x for x in self.inputs if x["type"] not in ["em", "es"]]

    def action(
        self,
        forecast: Action,
        climatology: Action,
        preprocessing_dim="param",
        ensemble_dim="number",
    ) -> Action:
        clim_headers = climatology.select({"type": "em"}, drop=True).map(
            Payload(grib.anomaly_clim)
        )
        action = forecast
        for preprocessing in self.preprocessing:
            action = action.param_operation(dim=preprocessing_dim, **preprocessing)
        action = action.anomaly(
            climatology.select({"type": "em"}, drop=True),
            climatology.select({"type": "es"}, drop=True),
            self.accumulations["step"].get("std_anomaly", False),
        )
        for dim, accumulation in self.accumulations.items():
            action = action.accum_operation(
                accumulation["operation"],
                dim=dim,
                coords=[accumulation["coords"]],
                metadata=accumulation.get("metadata", None),
                include_start=accumulation.get("include_start", False),
                deaccumulate=accumulation.get("deaccumulate", False),
            )
        return action.ensemble_operation(
            dim=ensemble_dim,
            **self.stats,
        )


def _translate_accum_op(accum: dict) -> str:
    OPS = {
        "aggregation": None,
        "difference": "diff",
        "maximum": "max",
        "minimum": "min",
        "mean": "mean",
        "standard_deviation": "std",
        "sum": "add",
    }
    operation = accum.setdefault("operation", "aggregation")
    if operation not in OPS:
        raise ValueError(f"Accumulation operation {operation} not supported")
    return OPS[operation]


def derive_config(request: dict, schema_config: dict) -> dict:
    ensemble_operation = {
        "em": "mean",
        "es": "std",
        "pb": "quantiles",
        "ep": "threshold_prob",
        "fcmean": None,
        "fcstdev": None,
        "fcmin": None,
        "fcmax": None,
        "efi": "efi",
        "efic": "efi",
        "sot": "sot",
        "cf": None,
        "pf": None,
    }
    schema_config.pop("entrypoint")
    schema_config.pop("interp_keys")
    schema_config.pop("name", None)
    schema_config.pop("dtype", None)
    inputs = schema_config.pop("inputs")

    # Populate coords in accumulations with values from inputs
    accums = schema_config.pop("accumulations", {})
    for dim, accum in accums.items():
        accum["operation"] = _translate_accum_op(accum)
        values = inputs[0][dim]
        accum["coords"] = [values] if isinstance(values, (str, int)) else values

    config = {
        "inputs": inputs,
        "preprocessing": schema_config.pop("preprocessing", []),
        "accumulations": accums,
        "stats": {
            "operation": ensemble_operation[request["type"]],
            **schema_config.pop("threshold", {}),
            **schema_config,
        },
    }
    config["stats"].setdefault("metadata", {}).setdefault("type", request["type"])
    return config


def derive_template(request: dict, pproc_schema: str) -> Action:
    schema = Schema.from_file(pproc_schema)
    schema_config = schema.config_from_output(request)
    templates = {
        "pproc-ensms": EnsembleConfig,
        "pproc-quantiles": EnsembleConfig,
        "pproc-probabilities": EnsembleConfig,
        "pproc-accumulate": EnsembleConfig,
        "pproc-extreme": ExtremeConfig,
        "pproc-anomaly-probabilities": AnomalyConfig,
        "pproc-wind": EnsembleConfig,
    }
    entrypoint = schema_config["entrypoint"]
    if entrypoint not in templates:
        raise ValueError(f"Schema {entrypoint} not supported")
    config = derive_config(request, schema_config)
    return templates[entrypoint](**config)


def from_request(request: dict, pproc_schema: str, **sources: Action) -> Action:
    config = derive_template(request, pproc_schema)
    return config.action(**sources)
