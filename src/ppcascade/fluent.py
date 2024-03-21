import numpy as np
import xarray as xr
import itertools

from cascade import fluent
from cascade import backends

from .utils.window import Range
from .utils.request import Request, MultiSourceRequest


class Action(fluent.Action):
    def non_descript_dim(self, dim: str):
        """
        Mark in node attributes a dimension in node array that is not
        a metadata key
        """
        self.nodes.attrs.setdefault("grib_exclude", set())
        self.nodes.attrs["grib_exclude"].add(dim)

    def non_descript_dims(self):
        return self.nodes.attrs.pop("grib_exclude", [])

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
        efi = concat.efi(climatology, windows, eps, window_dim)
        efi._add_dimension(new_dim, "efi")
        if efi_control is not None:
            control = self.select(efi_control, drop=True).efi(
                climatology, windows, eps, window_dim
            )
            control._add_dimension(new_dim, "efic")
            efi = efi.join(control, new_dim)
        sot = concat.sot(climatology, windows, eps, sot, window_dim, new_dim)
        ret = efi.join(sot, new_dim)
        ret.non_descript_dim(new_dim)
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
            )
            return self.join(climatology, "**datatype**").reduce(payload)

        join = self.join(climatology, "**datatype**", match_coord_values=True)
        return join.transform(
            _efi_window_transform,
            [({dim: window.name}, eps) for window in windows],
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
                _sot_transform, [(int(num), eps, new_dim) for num in sot], new_dim
            )
        else:
            params = [({dim: window.name}, sot, eps, new_dim) for window in windows]
            ret = self.join(
                climatology, "**datatype**", match_coord_values=True
            ).transform(_sot_window_transform, params, dim)
        ret.non_descript_dim(new_dim)
        return ret

    def ensms(
        self,
        dim: str = "number",
        new_dim: str | xr.DataArray = xr.DataArray(["mean", "std"], name="type"),
        batch_size: int = 0,
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
        mean = self.mean(dim, batch_size)
        std = self.std(dim, batch_size)
        res = mean.join(std, new_dim)
        res.non_descript_dim(new_dim if isinstance(new_dim, str) else new_dim.name)
        return res

    def threshold_prob(
        self,
        comparison: str,
        value: float,
        local_scale_factor: float = None,
        edition: int = 1,  # TODO: set this properly
        dim: str = "number",
        batch_size: int = 0,
    ):
        payload = fluent.Payload(
            backends.threshold,
            (
                fluent.Node.input_name(0),
                comparison,
                float(value),
                float(local_scale_factor) if local_scale_factor is not None else None,
                edition,
            ),
        )
        return self.map(payload).multiply(100).mean(dim, batch_size)

    def anomaly(
        self,
        clim_mean: fluent.Action,
        clim_std: fluent.Action,
        std_anomaly: bool = False,
        **method_kwargs,
    ):
        anom = self.subtract(clim_mean, **method_kwargs)
        if not std_anomaly:
            return anom
        return anom.divide(clim_std, **method_kwargs)

    def quantiles(
        self, num_quantiles: int = 100, dim: str = "number", new_dim: str = "quantile"
    ):
        params = [(x, new_dim) for x in np.linspace(0.0, 1.0, int(num_quantiles) + 1)]
        ret = self.concatenate(dim).transform(_quantiles_transform, params, new_dim)
        ret.non_descript_dim(new_dim)
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
                return getattr(self, operation)(dim=dim, **kwargs)
            operation = fluent.Payload(getattr(backends, operation), kwargs=kwargs)
        return self.reduce(operation, dim, batch_size)

    def window_operation(
        self,
        operation: str | fluent.Payload,
        ranges: list[Range],
        dim: str = "step",
        batch_size: int = 0,
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

        params = [(dim, range, operation, batch_size) for range in ranges]
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

    def write(self, target, config_grib_sets: dict):
        coords = list(self.nodes.dims)
        exclude = self.non_descript_dims()
        for node_attrs in itertools.product(
            *[self.nodes.coords[key].data for key in coords]
        ):
            node_coords = {key: node_attrs[index] for index, key in enumerate(coords)}
            node = self.node(node_coords)

            grib_sets = config_grib_sets.copy()
            grib_sets.update(self.nodes.attrs)
            grib_sets.update({k: v for k, v in node_coords.items() if k not in exclude})
            self.sinks.append(
                fluent.Node(
                    fluent.Payload(
                        backends.write, (target, fluent.Node.input_name(0), grib_sets)
                    ),
                    node,
                )
            )
        return self


def _sot_transform(
    action: fluent.Action, number: int, eps: float, new_dim: str
) -> fluent.Action:
    new_sot = action.reduce(
        fluent.Payload(
            backends.sot,
            (fluent.Node.input_name(1), fluent.Node.input_name(0), number, eps),
        )
    )
    new_sot._add_dimension(new_dim, number)
    return new_sot


def _sot_window_transform(
    action: fluent.Action, selection: dict, sot: list[int], eps: float, new_dim: str
) -> fluent.Action:
    return action.select(selection).transform(
        _sot_transform, [(int(num), eps, new_dim) for num in sot], new_dim
    )


def _efi_window_transform(
    action: fluent.Action, selection: dict, eps: float
) -> fluent.Action:
    ret = action.select(selection).reduce(
        fluent.Payload(
            backends.efi,
            (fluent.Node.input_name(1), fluent.Node.input_name(0), eps),
        ),
        dim="**datatype**",
    )
    return ret


def _quantiles_transform(action, quantile: float, new_dim: str):
    payload = fluent.Payload(backends.quantiles, (fluent.Node.input_name(0), quantile))
    new_quantile = action.map(payload)
    new_quantile._add_dimension(new_dim, quantile)
    return new_quantile


def _window_transform(
    action: fluent.Action, dim: str, range: Range, operation: str, batch_size: int
) -> fluent.Action:
    window_action = action.select({dim: range.steps})
    if len(range.steps) == 1:
        # Nothing to reduce
        return window_action

    if isinstance(operation, str):
        if hasattr(window_action, operation):
            window_action = getattr(window_action, operation)(
                dim=dim, batch_size=batch_size
            )
        else:
            operation = fluent.Payload(getattr(backends, operation))
            window_action = window_action.reduce(operation, dim, batch_size=batch_size)
    else:
        window_action = window_action.reduce(operation, dim, batch_size=batch_size)

    window_action._add_dimension(dim, range.name)
    return window_action


def _attribution_transform(action, config, scenario):
    payload = fluent.Payload(
        backends.attribution,
        (config, scenario, fluent.Node.input_name(0), fluent.Node.input_name(1)),
    )
    attr = action.reduce(payload)
    attr._add_dimension("scenario", scenario)
    return attr


from .backends.fieldlist import NumpyFieldListBackend


class PProcFluent(fluent.Fluent):
    def __init__(
        self,
        action=Action,
    ):
        super().__init__(action)

    def source(
        self,
        requests: list[Request | MultiSourceRequest],
        join_key: str = "",
        backend=NumpyFieldListBackend,
        **kwargs,
    ):
        all_actions = None
        for request in requests:
            payloads = np.empty(tuple(request.dims.values()), dtype=object)
            for indices, new_request in request.expand():
                payloads[indices] = (new_request,)
            new_action = super().source(
                backend.retrieve,
                xr.DataArray(
                    payloads,
                    coords={key: list(request[key]) for key in request.dims.keys()},
                ),
                kwargs,
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
