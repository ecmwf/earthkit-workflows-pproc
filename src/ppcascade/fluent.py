import functools
import inspect
from typing import Any

import numpy as np
from earthkit.workflows import fluent
from earthkit.workflows.backends.earthkit import FieldListBackend

from ppcascade.utils import grib, io, math
from ppcascade.utils.request import MultiSourceRequest, Request


class Action(fluent.Action):
    _THERMAL_CONFIG = {
        "utci": {"operation": math.calc_utci, "params": ["2t", "2d", "10si", "mrt"]},
        "10si": {"operation": "norm", "params": ["10u", "10v"]},
        "mrt": {
            "operation": math.calc_mrt,
            "params": ["uvcossza", "dsrp", "ssrd", "fdir", "strd", "str", "ssr"],
        },
        "uvcossza": {"operation": math.calc_cossza, "params": ["2t", "fdir"]},
        "dsrp": {"operation": math.calc_dsrp, "params": ["fdir", "uvcossza"]},
        "hmdx": {"operation": math.calc_hmdx, "params": ["2t", "2d"]},
        "2r": {"operation": math.calc_rhp, "params": ["2t", "2d"]},
        "heatx": {"operation": math.calc_heatx, "params": ["2t", "2d"]},
        "wbgt": {"operation": math.calc_wbgt, "params": ["2t", "2d", "10si", "mrt"]},
        "gt": {"operation": math.calc_gt, "params": ["2t", "10si", "mrt"]},
        "nefft": {"operation": math.calc_nefft, "params": ["2t", "10si", "2r"]},
        "wcf": {"operation": math.calc_wcf, "params": ["2t", "10si"]},
        "aptmp": {"operation": math.calc_aptmp, "params": ["2t", "2r", "10si"]},
    }

    def _reduction_with_metadata(
        self,
        operation: str,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        metadata: dict | None = None,
    ) -> "Action":
        batched = batch_size > 1 and self.nodes.sizes[dim] > batch_size
        op = getattr(super(), operation)
        if not batched or metadata is None:
            return op(
                dim=dim,
                batch_size=batch_size,
                keep_dim=keep_dim,
                backend_kwargs={"metadata": metadata},
            )

        # If batched, add additional node for setting window operation metadata. Doing this in a separate tasks
        # allows batched operations for overlapping windows to be identified and only computed once
        batched_action = op(dim=dim, batch_size=batch_size, keep_dim=keep_dim)
        return batched_action.map(
            fluent.Payload(
                FieldListBackend.set_metadata, [fluent.Node.input_name(0), metadata]
            )
        )

    def sum(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        metadata: dict | None = None,
    ) -> "Action":
        return self._reduction_with_metadata(
            "sum", dim=dim, batch_size=batch_size, keep_dim=keep_dim, metadata=metadata
        )

    def mean(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        metadata: dict | None = None,
    ) -> "Action":
        return self._reduction_with_metadata(
            "mean", dim=dim, batch_size=batch_size, keep_dim=keep_dim, metadata=metadata
        )

    def std(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        metadata: dict | None = None,
    ) -> "Action":
        return self._reduction_with_metadata(
            "std", dim=dim, batch_size=batch_size, keep_dim=keep_dim, metadata=metadata
        )

    def max(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        metadata: dict | None = None,
    ) -> "Action":
        return self._reduction_with_metadata(
            "max", dim=dim, batch_size=batch_size, keep_dim=keep_dim, metadata=metadata
        )

    maximum = max

    def min(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        metadata: dict | None = None,
    ) -> "Action":
        return self._reduction_with_metadata(
            "min", dim=dim, batch_size=batch_size, keep_dim=keep_dim, metadata=metadata
        )

    minimum = min

    def prod(
        self,
        dim: str = "",
        batch_size: int = 0,
        keep_dim: bool = False,
        metadata: dict | None = None,
    ) -> "Action":
        return self._reduction_with_metadata(
            "prod", dim=dim, batch_size=batch_size, keep_dim=keep_dim, metadata=metadata
        )

    def subtract(
        self, other: "Action | float", metadata: dict | None = None
    ) -> "Action":
        return super().subtract(other, backend_kwargs={"metadata": metadata})

    def divide(self, other: "Action | float", metadata: dict | None = None) -> "Action":
        return super().divide(other, backend_kwargs={"metadata": metadata})

    def add(self, other: "Action | float", metadata: dict | None = None) -> "Action":
        return super().add(other, backend_kwargs={"metadata": metadata})

    def multiply(
        self, other: "Action | float", metadata: dict | None = None
    ) -> "Action":
        return super().multiply(other, backend_kwargs={"metadata": metadata})

    def power(self, other: "Action | float", metadata: dict | None = None) -> "Action":
        return super().power(other, backend_kwargs={"metadata": metadata})

    def extreme(
        self,
        climatology: fluent.Action,
        step_ranges: list[str],
        sot: list[int],
        eps: float,
        efi_control: dict | None = None,
        ensemble_dim: str = "number",
        step_dim: str = "step",
        new_dim: str = "type",
        metadata: dict | None = None,
    ) -> "Action":
        """
        Create nodes computing the EFI and SOT

        Parameters
        ----------
        climatology: Action, nodes containing climatology data
        step_ranges: list of str, list of step ranges
        sot: list of ints, Shift-Of-Tail values
        eps: float
        efi_control: dict, selection for control member. If None, efi is not
        computed for the control member
        ensemble_dim: str, name of dimension for ensemble members
        step_dim: str, name of dimension for steps
        new_dim: str, name of new dimension corresponding to EFI/SOT nodes.

        Return
        ------
        Action
        """
        eps = float(eps)
        concat = self.concatenate(dim=ensemble_dim)
        efi = concat.efi(climatology, step_ranges, eps, step_dim, metadata)
        efi._add_dimension(new_dim, "efi")
        if efi_control is not None:
            control = self.select(efi_control, drop=True).efi(
                climatology, step_ranges, eps, step_dim, metadata
            )
            control._add_dimension(new_dim, "efic")
            efi = efi.join(control, new_dim)
        sot = concat.sot(
            climatology, step_ranges, eps, sot, step_dim, new_dim, metadata
        )
        ret = efi.join(sot, dim=new_dim)
        return ret

    def ensemble_extreme(
        self,
        operation: str,
        climatology: fluent.Action,
        step_ranges: list[str],
        ensemble_dim: str = "number",
        step_dim: str = "step",
        **kwargs,
    ) -> "Action":
        if operation == "extreme":
            return self.extreme(
                climatology,
                step_ranges,
                ensemble_dim=ensemble_dim,
                step_dim=step_dim,
                **kwargs,
            )
        return self.concatenate(ensemble_dim).__getattribute__(operation)(
            climatology, step_ranges, dim=step_dim, **kwargs
        )

    def efi(
        self,
        climatology: fluent.Action,
        step_ranges: list[str],
        eps: float,
        dim: str = "step",
        metadata: dict | None = None,
    ) -> "Action":
        """
        Create nodes computing the EFI for each window. Expects ensemble member dimension
        to already be concatenated into a single array.

        Parameters
        ----------
        climatology: Action, nodes containing climatology data
        step_ranges: list of str, list of step ranges
        eps: float
        dim: str, window dimension

        Return
        ------
        Action
        """
        eps = float(eps)
        if self.nodes.size == 1:
            if len(step_ranges) != 1:
                raise ValueError("Single node, but multiple step ranges")
            payload = fluent.Payload(
                math.efi,
                (fluent.Node.input_name(1), fluent.Node.input_name(0), eps),
                {"metadata": metadata},
            )
            return self.join(climatology, "**datatype**").reduce(payload)

        join = self.join(climatology, "**datatype**", match_coord_values=True)
        return join.transform(
            _efi_window_transform,
            [({dim: srange}, eps, metadata) for srange in step_ranges],
            dim,
        )

    def sot(
        self,
        climatology: fluent.Action,
        step_ranges: list[str],
        eps: float,
        sot: list[int],
        dim: str = "step",
        new_dim: str = "sot",
        metadata: dict | None = None,
    ) -> "Action":
        """
        Create nodes computing the SOT for each window. Expects ensemble member dimension
        to already be concatenated into a single array.

        Parameters
        ----------
        climatology: Action, nodes containing climatology data
        step_ranges: list of str, list of step ranges
        eps: float
        sot: list of ints, Shift-Of-Tail values
        dim: str, window dimension
        new_dim: str, name of new dimension corresponding to SOT nodes.

        Return
        ------
        Action
        """
        eps = float(eps)
        if not isinstance(sot, list):
            sot = [sot]

        if self.nodes.size == 1:
            if len(step_ranges) != 1:
                raise ValueError("Single node, but multiple step ranges")
            ret = self.join(
                climatology, "**datatype**", match_coord_values=True
            ).transform(
                _sot_transform,
                [(int(num), eps, new_dim, metadata) for num in sot],
                new_dim,
            )
        else:
            params = [
                ({dim: srange}, sot, eps, new_dim, metadata) for srange in step_ranges
            ]
            ret = self.join(
                climatology, "**datatype**", match_coord_values=True
            ).transform(_sot_window_transform, params, dim)
        return ret

    def ensms(
        self,
        dim: str = "number",
        new_dim: str | tuple[str, list] = ("type", ["mean", "std"]),
        batch_size: int = 0,
        metadata: dict | None = None,
    ) -> "Action":
        """
        Creates nodes computing the mean and standard deviation along the specified dimension. A new dimension
        is created joining the mean and standard deviation. If batch_size > 1 and less than the size
        of the named dimension, the reduction will be computed first in
        batches and then aggregated, otherwise no batching will be performed.

        Parameters
        ----------
        dim: str, dimension to compute mean and standard deviation along
        new_dim: str or tuple, name of new dimension or tuple specifying new dimension name and
        coordinate values
        batch_size: int, size of batches to split reduction into. If 0,
        computation is not batched

        Return
        ------
        Action
        """
        if metadata is None:
            metadata = {}
        mean = self.mean(
            dim=dim, batch_size=batch_size, metadata={"type": "em", **metadata}
        )
        std = self.std(
            dim=dim, batch_size=batch_size, metadata={"type": "es", **metadata}
        )
        res = mean.join(std, new_dim)
        return res

    def threshold_prob(
        self,
        comparison: str,
        value: float,
        local_scale_factor: float = None,
        dim: str = "number",
        batch_size: int = 0,
        metadata: dict | None = None,
    ) -> "Action":
        if metadata is not None and int(metadata.get("edition", 1)) == 1:
            metadata = metadata.copy()
            metadata.update(
                grib.threshold(
                    comparison,
                    float(value),
                    (
                        float(local_scale_factor)
                        if local_scale_factor is not None
                        else None
                    ),
                )
            )
        payload = fluent.Payload(
            math.threshold,
            (
                fluent.Node.input_name(0),
                comparison,
                float(value),
            ),
        )
        return (
            self.map(payload)
            .multiply(100)
            .mean(dim=dim, batch_size=batch_size, metadata=metadata)
        )

    def anomaly(
        self,
        clim_mean: fluent.Action,
        clim_std: fluent.Action,
        std_anomaly: bool = False,
        metadata: dict | None = None,
    ) -> "Action":
        anom = self.subtract(clim_mean, metadata=metadata)
        if not std_anomaly:
            return anom
        return anom.divide(clim_std)

    def quantiles(
        self,
        quantiles: int | list[float],
        dim: str = "number",
        new_dim: str = "quantile",
        metadata: dict | None = None,
    ) -> "Action":
        quantiles = (
            quantiles
            if isinstance(quantiles, list)
            else np.linspace(0.0, 1.0, int(quantiles) + 1)
        )
        params = [(x, new_dim, metadata) for x in quantiles]
        ret = self.concatenate(dim).transform(_quantiles_transform, params, new_dim)
        return ret

    def wind_speed(
        self, vod2uv: bool, dim: str = "param", metadata: dict | None = None
    ) -> "Action":
        kwargs = {"metadata": metadata}
        if vod2uv:
            ret = self.map(
                fluent.Payload(
                    FieldListBackend.norm, (fluent.Node.input_name(0),), kwargs
                )
            )
        else:
            ret = self.reduce(
                fluent.Payload(FieldListBackend.norm, kwargs=kwargs), dim=dim
            )
        return ret

    def _wrapped_reduction(
        self,
        operation: str | fluent.Payload | None,
        dim: str,
        batch_size: int = 0,
        metadata: dict | None = None,
        **kwargs,
    ) -> "Action":
        if operation is None:
            return self
        if isinstance(operation, str):
            if hasattr(self, operation):
                op = getattr(self, operation)
                sig = inspect.signature(op)
                args = [
                    p.name
                    for p in sig.parameters.values()
                    if p.kind == p.POSITIONAL_OR_KEYWORD
                ]
                if "batch_size" in args:
                    kwargs["batch_size"] = batch_size
                return op(dim=dim, metadata=metadata, **kwargs)
            if metadata is not None:
                kwargs.setdefault("metadata", {}).update(metadata)
            op_function = getattr(FieldListBackend, operation, None) or getattr(
                math, operation, None
            )
            if op_function is None:
                raise ValueError(f"Operation {operation} not found")
            operation = fluent.Payload(
                op_function,
                kwargs=kwargs,
            )
        return self.reduce(operation, dim=dim, batch_size=batch_size)

    def param_operation(
        self,
        operation: str | fluent.Payload | None,
        dim: str = "param",
        batch_size: int = 0,
        metadata: dict | None = None,
        **kwargs,
    ) -> "Action":
        """
        Reduction operation across different parameters

        Params
        ------
        operation: str or Payload, operation to perform on ensemble members
        dim: str, dimension to perform operation along

        Return
        ------
        Action
        """
        if "operation" == "scale":
            return self.multiply(kwargs["value"], metadata=metadata)
        return self._wrapped_reduction(operation, dim, batch_size, metadata, **kwargs)

    def thermal_index(
        self, param: str, dim: str = "param", metadata: dict | None = None
    ) -> "Action":
        config = self._THERMAL_CONFIG[param]
        for input in config["params"]:
            if input not in self.nodes.coords[dim] and input in self._THERMAL_CONFIG:
                dependency = self.thermal_index(input, dim=dim, metadata=metadata)
                dependency._add_dimension(dim, input)
                self = self.join(dependency, dim)
        try:
            selection = self.sel(param=(config["params"]))
        except KeyError:
            selection = self
        ret = selection._wrapped_reduction(
            config["operation"], dim=dim, metadata=metadata
        )
        return ret

    def mask(
        self,
        select: dict,
        mask: tuple[dict, str, float],
        replacement: float = 0.0,
        dim: str = "param",
        metadata: dict | None = None,
    ) -> "Action":
        filter_selection, comparison, threshold = mask
        selected_nodes = self.sel(select).join(self.sel(filter_selection), dim=dim)
        return selected_nodes._wrapped_reduction(
            "filter",
            dim=dim,
            metadata=metadata,
            comparison=comparison,
            threshold=threshold,
            replacement=replacement,
        )

    def ensemble_operation(
        self,
        operation: str | fluent.Payload | None,
        dim: str = "number",
        batch_size: int = 0,
        metadata: dict | None = None,
        **kwargs,
    ) -> "Action":
        """
        Reduction operation across ensemble members. If batch_size > 1 and less than the size
        of the named dimension, the reduction will be computed first in
        batches and then aggregated, otherwise no batching will be performed.

        Params
        ------
        operation: str or Payload, operation to perform on ensemble members
        dim: str, dimension to perform operation along
        batch_size: int, size of batches to split reduction into. If 0,
        computation is not batched

        Return
        ------
        Action

        Raises
        ------
        ValueError if payload function is not batchable and batch_size is not 0
        """
        if operation in ["mean", "std"]:
            metadata = {} if metadata is None else metadata.copy()
            if "type" not in metadata:
                metadata["type"] = "em" if operation == "mean" else "es"
        return self._wrapped_reduction(operation, dim, batch_size, metadata, **kwargs)

    def accum_operation(
        self,
        operation: str | fluent.Payload,
        coords: list[list[Any]],
        dim: str = "step",
        batch_size: int = 0,
        metadata: dict | None = None,
        include_start: bool = False,
        deaccumulate: bool = False,
        **kwargs,
    ) -> "Action":
        """
        Reduction operation across a dimension. If batch_size > 1 and less than the size
        of the named dimension, the reduction will be computed first in
        batches and then aggregated, otherwise no batching will be performed.

        Params
        ------
        operation: str or Payload, operation to perform on steps
        coords: list of values to accumulate over
        dim: str, dimension to perform operation along
        batch_size: int, size of batches to split reduction into. If 0,
        computation is not batched
        metadata: optional dict, metadata to set on the output
        include_init: bool, whether to include the initial value in the accumulation
        deaccumulate: bool, whether to deaccumulate consecutive values before accumulation

        Return
        ------
        Action

        Raises
        ------
        ValueError if payload function is not batchable and batch_size is not 0
        """
        if operation is None:
            self._squeeze_dimension(dim)
            return self

        params = [
            (
                dim,
                coord,
                operation,
                batch_size,
                metadata,
                include_start,
                deaccumulate,
                kwargs,
            )
            for coord in coords
        ]
        return self.transform(_accum_transform, params, dim)

    def write(
        self, target: str, metadata: dict | fluent.Action | None = None
    ) -> "Action":
        if isinstance(metadata, fluent.Action):
            res = self.join(metadata, "**datatype**", match_coord_values=True).reduce(
                fluent.Payload(
                    io.write,
                    (fluent.Node.input_name(0), target, fluent.Node.input_name(1)),
                ),
                dim="**datatype**",
            )
            return res

        return self.map(
            fluent.Payload(
                io.write,
                (fluent.Node.input_name(0), target),
                {"metadata": metadata},
            )
        )


def _sot_transform(
    action: fluent.Action, number: int, eps: float, new_dim: str, metadata: dict | None
) -> fluent.Action:
    new_sot = action.reduce(
        fluent.Payload(
            math.sot,
            (fluent.Node.input_name(1), fluent.Node.input_name(0), number, eps),
            {"metadata": metadata},
        )
    )
    new_sot._add_dimension(new_dim, number)
    return new_sot


def _sot_window_transform(
    action: fluent.Action,
    selection: dict,
    sot: list[int],
    eps: float,
    new_dim: str,
    metadata: dict,
) -> fluent.Action:
    return action.select(selection).transform(
        _sot_transform, [(int(num), eps, new_dim, metadata) for num in sot], new_dim
    )


def _efi_window_transform(
    action: fluent.Action, selection: dict, eps: float, metadata: dict | None
) -> fluent.Action:
    ret = action.select(selection).reduce(
        fluent.Payload(
            math.efi,
            (fluent.Node.input_name(1), fluent.Node.input_name(0), eps),
            {"metadata": metadata},
        ),
        dim="**datatype**",
    )
    return ret


def _quantiles_transform(action, quantile: float, new_dim: str, metadata: dict | None):
    payload = fluent.Payload(
        math.quantiles,
        (fluent.Node.input_name(0), quantile),
        {"metadata": metadata},
    )
    new_quantile = action.map(payload)
    new_quantile._add_dimension(new_dim, quantile)
    return new_quantile


def _accum_transform(
    action: fluent.Action,
    dim: str,
    coords: list[Any],
    operation: str | fluent.Payload,
    batch_size: int,
    metadata: dict | None,
    include_start: bool,
    deaccumulate: bool,
    kwargs: dict,
) -> fluent.Action:
    if len(coords) == 1:
        # Nothing to reduce and no metadata to set
        return action.select({dim: coords})

    metadata = {} if metadata is None else metadata.copy()
    if dim == "step":
        metadata.update(grib.window(operation, coords, include_start))

    if deaccumulate:
        accum_action = action.select({dim: coords[:-1]})
        accum_action = accum_action.subtract(action.select({dim: coords[1:]}))
    else:
        accum_action = action.select({dim: coords if include_start else coords[1:]})

    accum_action = accum_action._wrapped_reduction(
        operation,
        dim=dim,
        batch_size=batch_size,
        metadata=metadata,
        **kwargs,
    )

    accum_action._add_dimension(dim, f"{coords[0]}-{coords[-1]}")
    return accum_action


def from_source(
    requests: list[dict | Request | MultiSourceRequest],
    join_key: str = "",
    backend_kwargs: dict = {},
):
    all_actions = None
    for request in requests:
        if isinstance(request, dict):
            request = Request(request)
        payloads = np.empty(tuple(request.dims.values()), dtype=object)
        for indices, new_request in request.expand():
            payloads[indices] = functools.partial(
                io.retrieve, new_request, **backend_kwargs
            )
        new_action = fluent.from_source(
            payloads,
            coords={key: list(request[key]) for key in request.dims.keys()},
            action=Action,
        )

        if len(join_key) != 0 and join_key not in new_action.nodes.coords:
            new_action._add_dimension(join_key, request[join_key])

        if all_actions is None:
            all_actions = new_action
        else:
            if len(join_key) == 0:
                raise ValueError("Join key must be specified for multiple requests")
            all_actions = all_actions.join(new_action, join_key)
    return all_actions
