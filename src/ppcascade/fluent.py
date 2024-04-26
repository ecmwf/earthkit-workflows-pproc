import numpy as np
import xarray as xr
import functools
import inspect

from cascade import fluent
from cascade import backends

from .utils.window import Range
from .utils.request import Request, MultiSourceRequest
from .utils import grib
from .backends import fieldlist


class Action(fluent.Action):
    def extreme(
        self,
        climatology: fluent.Action,
        windows: list[Range],
        sot: list[int],
        eps: float,
        efi_control: dict | None = None,
        ensemble_dim: str = "number",
        window_dim: str = "step",
        new_dim: str = "type",
        metadata: dict | None = None,
    ):
        """
        Create nodes computing the EFI and SOT

        Parameters
        ----------
        climatology: Action, nodes containing climatology data
        windows: list of Range, list of window ranges
        sot: list of ints, Shift-Of-Tail values
        eps: float
        efi_control: dict, selection for control member. If None, efi is not
        computed for the control member
        dim: str, name of dimension for ensemble members
        new_dim: str, name of new dimension corresponding to EFI/SOT nodes.

        Return
        ------
        MultiAction
        """
        eps = float(eps)
        concat = self.concatenate(ensemble_dim)
        efi = concat.efi(climatology, windows, eps, window_dim, metadata)
        efi._add_dimension(new_dim, "efi")
        if efi_control is not None:
            control = self.select(efi_control, drop=True).efi(
                climatology, windows, eps, window_dim, metadata
            )
            control._add_dimension(new_dim, "efic")
            efi = efi.join(control, new_dim)
        sot = concat.sot(climatology, windows, eps, sot, window_dim, new_dim, metadata)
        ret = efi.join(sot, new_dim)
        return ret

    def ensemble_extreme(
        self,
        operation: str,
        climatology: fluent.Action,
        windows: list[Range],
        ensemble_dim: str = "number",
        window_dim: str = "step",
        **kwargs,
    ):
        if operation == "extreme":
            return self.extreme(
                climatology,
                windows,
                ensemble_dim=ensemble_dim,
                window_dim=window_dim,
                **kwargs,
            )
        return self.concatenate(ensemble_dim).__getattribute__(operation)(
            climatology, windows, dim=window_dim, **kwargs
        )

    def efi(
        self,
        climatology: fluent.Action,
        windows: list[Range],
        eps: float,
        dim: str = "step",
        metadata: dict | None = None,
    ):
        """
        Create nodes computing the EFI for each window. Expects ensemble member dimension
        to already be concatenated into a single array.

        Parameters
        ----------
        climatology: Action, nodes containing climatology data
        windows: list of Range, list of window ranges
        eps: float
        dim: str, window dimension

        Return
        ------
        MultiAction
        """
        eps = float(eps)
        if self.nodes.size == 1:
            if len(windows) != 1:
                raise ValueError("Single node, but multiple windows")
            payload = fluent.Payload(
                backends.efi,
                (fluent.Node.input_name(1), fluent.Node.input_name(0), eps),
                {"metadata": metadata} if metadata is not None else {},
            )
            return self.join(climatology, "**datatype**").reduce(payload)

        join = self.join(climatology, "**datatype**", match_coord_values=True)
        return join.transform(
            _efi_window_transform,
            [({dim: window.name}, eps, metadata) for window in windows],
            dim,
        )

    def sot(
        self,
        climatology: fluent.Action,
        windows: list[Range],
        eps: float,
        sot: list[int],
        dim: str = "step",
        new_dim: str = "sot",
        metadata: dict | None = None,
    ):
        """
        Create nodes computing the SOT for each window. Expects ensemble member dimension
        to already be concatenated into a single array.

        Parameters
        ----------
        climatology: Action, nodes containing climatology data
        windows: list of Range, list of window ranges
        eps: float
        sot: list of ints, Shift-Of-Tail values
        dim: str, window dimension
        new_dim: str, name of new dimension corresponding to SOT nodes.

        Return
        ------
        MultiAction
        """
        eps = float(eps)
        if not isinstance(sot, list):
            sot = [sot]

        if self.nodes.size == 1:
            if len(windows) != 1:
                raise ValueError("Single node, but multiple windows")
            ret = self.join(
                climatology, "**datatype**", match_coord_values=True
            ).transform(
                _sot_transform,
                [(int(num), eps, new_dim, metadata) for num in sot],
                new_dim,
            )
        else:
            params = [
                ({dim: window.name}, sot, eps, new_dim, metadata) for window in windows
            ]
            ret = self.join(
                climatology, "**datatype**", match_coord_values=True
            ).transform(_sot_window_transform, params, dim)
        return ret

    def ensms(
        self,
        dim: str = "number",
        new_dim: str | xr.DataArray = xr.DataArray(["mean", "std"], name="type"),
        batch_size: int = 0,
        metadata: dict | None = None,
    ):
        """
        Creates nodes computing the mean and standard deviation along the specified dimension. A new dimension
        is created joining the mean and standard deviation. If batch_size > 1 and less than the size
        of the named dimension, the reduction will be computed first in
        batches and then aggregated, otherwise no batching will be performed.

        Parameters
        ----------
        dim: str, dimension to compute mean and standard deviation along
        new_dim: str or xr.DataArray, name of new dimension or xr.DataArray specifying new dimension name and
        coordinate values
        batch_size: int, size of batches to split reduction into. If 0,
        computation is not batched

        Return
        ------
        MultiAction
        """
        if metadata is None:
            metadata = {}
        mean = self.mean(dim, batch_size, metadata={"type": "em", **metadata})
        std = self.std(dim, batch_size, metadata={"type": "es", **metadata})
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
    ):
        if metadata is None:
            metadata = {}
        if int(metadata.get("edition", 1)) == 1:
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
            backends.threshold,
            (
                fluent.Node.input_name(0),
                comparison,
                float(value),
            ),
        )
        return self.map(payload).multiply(100).mean(dim, batch_size, metadata=metadata)

    def anomaly(
        self,
        clim_mean: fluent.Action,
        clim_std: fluent.Action,
        std_anomaly: bool = False,
        metadata: dict | None = None,
    ):
        anom = self.subtract(clim_mean, metadata=metadata)
        if not std_anomaly:
            return anom
        return anom.divide(clim_std)

    def quantiles(
        self,
        num_quantiles: int = 100,
        dim: str = "number",
        new_dim: str = "quantile",
        metadata: dict | None = None,
    ):
        params = [
            (x, new_dim, metadata)
            for x in np.linspace(0.0, 1.0, int(num_quantiles) + 1)
        ]
        ret = self.concatenate(dim).transform(_quantiles_transform, params, new_dim)
        return ret

    def wind_speed(self, vod2uv: bool, dim: str = "param"):
        if vod2uv:
            ret = self.map(fluent.Payload(backends.norm, (fluent.Node.input_name(0),)))
        else:
            ret = self.reduce(fluent.Payload(backends.norm), dim)
        return ret

    def param_operation(
        self, operation: str | fluent.Payload | None, dim: str = "param", **kwargs
    ):
        """
        Reduction operation across different parameters

        Params
        ------
        operation: str or Payload, operation to perform on ensemble members
        dim: str, dimension to perform operation along

        Return
        ------
        Single or MultiAction
        """
        if operation is None:
            return self
        if isinstance(operation, str):
            if hasattr(self, operation):
                return getattr(self, operation)(dim=dim, **kwargs)
            operation = fluent.Payload(getattr(backends, operation), kwargs=kwargs)
        return self.reduce(operation, dim)

    def ensemble_operation(
        self,
        operation: str | fluent.Payload | None,
        dim: str = "number",
        batch_size: int = 0,
        **kwargs,
    ):
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
        Single or MultiAction

        Raises
        ------
        ValueError if payload function is not batchable and batch_size is not 0
        """
        if operation is None:
            return self
        if isinstance(operation, str):
            if hasattr(self, operation):
                if operation == "mean":
                    kwargs.setdefault("metadata", {})
                    kwargs["metadata"] = {"type": "em"}
                elif operation == "std":
                    kwargs.setdefault("metadata", {})
                    kwargs["metadata"] = {"type": "es"}
                return getattr(self, operation)(dim=dim, **kwargs)
            operation = fluent.Payload(getattr(backends, operation), kwargs=kwargs)
        return self.reduce(operation, dim, batch_size)

    def window_operation(
        self,
        operation: str | fluent.Payload,
        ranges: list[Range],
        dim: str = "step",
        batch_size: int = 0,
        **kwargs,
    ):
        """
        Reduction operation across steps. If batch_size > 1 and less than the size
        of the named dimension, the reduction will be computed first in
        batches and then aggregated, otherwise no batching will be performed.

        Params
        ------
        operation: str or Payload, operation to perform on steps
        ranges: list of Range, window ranges
        dim: str, dimension to perform operation along
        batch_size: int, size of batches to split reduction into. If 0,
        computation is not batched

        Return
        ------
        Single or MultiAction

        Raises
        ------
        ValueError if payload function is not batchable and batch_size is not 0
        """
        if operation is None:
            self._squeeze_dimension(dim)
            return self

        params = [(dim, range, operation, batch_size, kwargs) for range in ranges]
        return self.transform(_window_transform, params, dim)

    def pca(self, config, mask, target: str = None):
        if mask is not None:
            raise NotImplementedError()
        return self.reduce(
            fluent.Payload(
                backends.pca,
                (
                    config,
                    fluent.Node.input_name(0),
                    fluent.Node.input_name(1),
                    mask,
                    target,
                ),
            )
        )

    def cluster(self, config, ncomp_file, indexes, deterministic):
        return self.map(
            fluent.Payload(
                backends.cluster,
                (config, fluent.Node.input_name(0), ncomp_file, indexes, deterministic),
            )
        )

    def attribution(self, config, targets):
        return self.transform(
            _attribution_transform,
            [(config, "centroids"), (config, "representatives")],
            "scenario",
        ).map(
            np.asarray(
                [
                    fluent.Payload(
                        backends.cluster_write,
                        (
                            config,
                            "centroids",
                            fluent.Node.input_name(0),
                            targets["centroids"],
                        ),
                    ),
                    fluent.Payload(
                        backends.cluster_write,
                        (
                            config,
                            "representatives",
                            fluent.Node.input_name(0),
                            targets["representatives"],
                        ),
                    ),
                ]
            )
        )

    def write(self, target: str, metadata: dict | fluent.Action | None = None):
        if isinstance(metadata, fluent.Action):
            res = self.join(metadata, "**datatype**", match_coord_values=True).reduce(
                fluent.Payload(
                    backends.write,
                    (fluent.Node.input_name(0), target, fluent.Node.input_name(1)),
                ),
                dim="**datatype**",
            )
            return res

        return self.map(
            fluent.Payload(
                backends.write,
                (fluent.Node.input_name(0), target),
                {"metadata": metadata},
            )
        )


def _sot_transform(
    action: fluent.Action, number: int, eps: float, new_dim: str, metadata: dict | None
) -> fluent.Action:
    if metadata is None:
        metadata = {}
    new_sot = action.reduce(
        fluent.Payload(
            backends.sot,
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
    if metadata is None:
        metadata = {}
    ret = action.select(selection).reduce(
        fluent.Payload(
            backends.efi,
            (fluent.Node.input_name(1), fluent.Node.input_name(0), eps),
            {"metadata": metadata},
        ),
        dim="**datatype**",
    )
    return ret


def _quantiles_transform(action, quantile: float, new_dim: str, metadata: dict | None):
    if metadata is None:
        metadata = {}
    payload = fluent.Payload(
        backends.quantiles,
        (fluent.Node.input_name(0), quantile),
        {"metadata": metadata},
    )
    new_quantile = action.map(payload)
    new_quantile._add_dimension(new_dim, quantile)
    return new_quantile


def _window_transform(
    action: fluent.Action,
    dim: str,
    range: Range,
    operation: str | fluent.Payload,
    batch_size: int,
    kwargs: dict,
) -> fluent.Action:
    window_action = action.select({dim: range.steps})
    if len(range.steps) == 1:
        # Nothing to reduce and no metadata to set
        return window_action

    batched = batch_size > 1 and len(range.steps) > batch_size
    window_metadata = grib.window(operation, range)
    if not batched:
        kwargs.setdefault("metadata", {}).update(window_metadata)
    else:
        metadata = kwargs.pop("metadata", {})
        metadata.update(window_metadata)
        window_metadata = metadata

    if isinstance(operation, str):
        if hasattr(window_action, operation):
            # Use method if it exists as there can be special batch handling, such as
            # for mean
            window_action = getattr(window_action, operation)(
                dim=dim, batch_size=batch_size, **kwargs
            )
        else:
            operation = fluent.Payload(
                getattr(backends, operation),
                kwargs=kwargs,
            )
            window_action = window_action.reduce(operation, dim, batch_size=batch_size)
    else:
        window_action = window_action.reduce(operation, dim, batch_size=batch_size)

    window_action._add_dimension(dim, range.name)
    if not batched:
        return window_action
    # If batched, add additional node for setting window operation metadata. Doing this in a separate tasks
    # allows batched operations for overlapping windows to be identified and only computed once
    return window_action.map(
        fluent.Payload(
            backends.set_metadata, [fluent.Node.input_name(0), window_metadata]
        )
    )


def _attribution_transform(action, config, scenario):
    payload = fluent.Payload(
        backends.attribution,
        (config, scenario, fluent.Node.input_name(0), fluent.Node.input_name(1)),
    )
    attr = action.reduce(payload)
    attr._add_dimension("scenario", scenario)
    return attr


def from_source(
    requests: list[dict | Request | MultiSourceRequest],
    join_key: str = "",
    backend=fieldlist.NumpyFieldListBackend,
    **kwargs,
):
    all_actions = None
    for request in requests:
        if isinstance(request, dict):
            request = Request(request)
        payloads = np.empty(tuple(request.dims.values()), dtype=object)
        for indices, new_request in request.expand():
            payloads[indices] = functools.partial(
                backend.retrieve, new_request, **kwargs
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
