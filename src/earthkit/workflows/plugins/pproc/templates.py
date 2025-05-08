# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import dataclass
from typing import Optional

from earthkit.workflows.fluent import Payload
from pproc.config.preprocessing import PreprocessingConfig
from pproc.schema.schema import Schema

from earthkit.workflows.plugins.pproc.fluent import Action
from earthkit.workflows.plugins.pproc.utils import grib


@dataclass
class StatsConfig:
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
            operation=self.stats.operation,
            metadata={**self.stats.metadata, **clim_headers},
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


def derive_config(
    request: dict, schema_config: dict, metadata: Optional[dict] = None
) -> dict:
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
    if metadata:
        config["stats"]["metadata"].update(metadata)
    return config


def derive_template(
    request: dict,
    pproc_schema: str,
    inputs: Optional[list[dict]] = None,
    metadata: Optional[dict] = None,
) -> Action:
    schema = Schema.from_file(pproc_schema)
    schema_config = schema.config_from_output(request, inputs=inputs)
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
    config = derive_config(request, schema_config, metadata)
    return templates[entrypoint](**config)


def from_request(
    request: dict,
    pproc_schema: str,
    preprocessing_dim: str = "param",
    ensemble_dim: str = "number",
    metadata: Optional[dict] = None,
    **sources: Action,
) -> Action:
    config = derive_template(request, pproc_schema, metadata=metadata)
    return config.action(
        **sources, preprocessing_dim=preprocessing_dim, ensemble_dim=ensemble_dim
    )
