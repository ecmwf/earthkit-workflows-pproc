import numpy as np
import xarray as xr
import itertools

from cascade.fluent import Action, Node, Payload
from cascade.fluent import SingleAction as BaseSingleAction
from cascade.fluent import MultiAction as BaseMultiAction
from cascade.fluent import Fluent
from cascade import backends


from .utils.window import Range
from .utils.request import Request, MultiSourceRequest


class SingleAction(BaseSingleAction):
    def to_multi(self, nodes: xr.DataArray):
        return MultiAction(self, nodes)

    def non_descript_dim(self, dim: str):
        """
        Mark in node attributes a dimension in node array that is not
        a metadata key
        """
        self.nodes.attrs.setdefault("grib_exclude", set())
        self.nodes.attrs["grib_exclude"].add(dim)

    def non_descript_dims(self):
        return self.nodes.attrs.pop("grib_exclude", [])

    def efi(
        self, climatology: Action, windows: list[Range], eps: float, dim: str = "step"
    ):
        eps = float(eps)
        assert len(windows) == 1, f"Only one window is supported, got {windows}"
        assert self.nodes.coords[dim] == windows[0].name
        # Join with climatology and compute efi
        payload = Payload(backends.efi, (Node.input_name(1), Node.input_name(0), eps))
        return self.join(climatology, "**datatype**").reduce(payload)

    def sot(
        self,
        climatology: Action,
        windows: list[Range],
        eps: float,
        sot: list[int],
        dim: str = "step",
        new_dim: str = "sot",
    ):
        eps = float(eps)
        assert len(windows) == 1
        assert self.nodes.coords[dim] == windows[0].name
        if not isinstance(sot, list):
            sot = [sot]

        ret = self.join(climatology, "**datatype**", match_coord_values=True).transform(
            _sot_transform, [(int(num), eps, new_dim) for num in sot], new_dim
        )
        ret.non_descript_dim(new_dim)
        return ret

    def cluster(self, config, ncomp_file, indexes, deterministic):
        return self.map(
            Payload(
                backends.cluster,
                (config, Node.input_name(0), ncomp_file, indexes, deterministic),
            )
        )

    def write(self, target, config_grib_sets: dict):
        grib_sets = config_grib_sets.copy()
        exclude = self.non_descript_dims()
        grib_sets.update(self.nodes.attrs)
        for name, values in self.nodes.coords.items():
            if name in exclude:
                continue
            if values.data.ndim == 0:
                grib_sets[name] = values.data
            else:
                assert values.data.ndim == 1
                grib_sets[name] = values.data[0]
        payload = Payload(backends.write, (target, Node.input_name(0), grib_sets))
        self.sinks.append(Node(payload, self.node()))
        return self


class MultiAction(BaseMultiAction):
    def to_single(self, payload_or_node: Payload | Node):
        if isinstance(payload_or_node, Payload):
            return SingleAction.from_payload(self, payload_or_node)
        return SingleAction(self, payload_or_node)

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
        climatology: Action,
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
        climatology: Action,
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
        self, climatology: Action, windows: list[Range], eps: float, dim: str = "step"
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
        return self.join(
            climatology, "**datatype**", match_coord_values=True
        ).transform(
            _efi_window_transform,
            [({dim: window.name}, eps) for window in windows],
            dim,
        )

    def sot(
        self,
        climatology: Action,
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

        params = [({dim: window.name}, sot, eps, new_dim) for window in windows]
        ret = self.join(climatology, "**datatype**", match_coord_values=True).transform(
            _sot_window_transform, params, dim
        )
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
        payload = Payload(
            backends.threshold,
            (
                Node.input_name(0),
                comparison,
                float(value),
                float(local_scale_factor) if local_scale_factor is not None else None,
                edition,
            ),
        )
        return self.map(payload).multiply(100).mean(dim, batch_size)

    def anomaly(
        self,
        clim_mean: Action,
        clim_std: Action,
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
            ret = self.map(Payload(backends.norm, (Node.input_name(0),)))
        else:
            ret = self.reduce(Payload(backends.norm), dim)
        return ret

    def param_operation(
        self, operation: str | Payload | None, dim: str = "param", **kwargs
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
            operation = Payload(getattr(backends, operation), kwargs=kwargs)
        return self.reduce(operation, dim)

    def ensemble_operation(
        self,
        operation: str | Payload | None,
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
            operation = Payload(getattr(backends, operation), kwargs=kwargs)
        return self.reduce(operation, dim, batch_size)

    def window_operation(
        self,
        operation: str | Payload,
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
            Payload(
                backends.pca,
                (config, Node.input_name(0), Node.input_name(1), mask, target),
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
                    Payload(
                        backends.cluster_write,
                        (config, "centroids", Node.input_name(0), targets["centroids"]),
                    ),
                    Payload(
                        backends.cluster_write,
                        (
                            config,
                            "representatives",
                            Node.input_name(0),
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
                Node(
                    Payload(backends.write, (target, Node.input_name(0), grib_sets)),
                    node,
                )
            )
        return self


def _sot_transform(action: Action, number: int, eps: float, new_dim: str) -> Action:
    new_sot = action.reduce(
        Payload(
            backends.sot,
            (Node.input_name(1), Node.input_name(0), number, eps),
        )
    )
    new_sot._add_dimension(new_dim, number)
    return new_sot


def _sot_window_transform(
    action: Action, selection: dict, sot: list[int], eps: float, new_dim: str
) -> Action:
    return action.select(selection).transform(
        _sot_transform, [(int(num), eps, new_dim) for num in sot], new_dim
    )


def _efi_window_transform(action: Action, selection: dict, eps: float) -> Action:
    ret = action.select(selection).reduce(
        Payload(
            backends.efi,
            (Node.input_name(1), Node.input_name(0), eps),
        ),
        dim="**datatype**",
    )
    return ret


def _quantiles_transform(action, quantile: float, new_dim: str):
    payload = Payload(backends.quantiles, (Node.input_name(0), quantile))
    new_quantile = action.map(payload)
    new_quantile._add_dimension(new_dim, quantile)
    return new_quantile


def _window_transform(
    action: Action, dim: str, range: Range, operation: str, batch_size: int
) -> Action:
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
            operation = Payload(getattr(backends, operation))
            window_action = window_action.reduce(operation, dim, batch_size=batch_size)
    else:
        window_action = window_action.reduce(operation, dim, batch_size=batch_size)

    window_action._add_dimension(dim, range.name)
    return window_action


def _attribution_transform(action, config, scenario):
    payload = Payload(
        backends.attribution,
        (config, scenario, Node.input_name(0), Node.input_name(1)),
    )
    attr = action.reduce(payload)
    attr._add_dimension("scenario", scenario)
    return attr


from .backends.fieldlist import NumpyFieldListBackend


class PProcFluent(Fluent):
    def __init__(
        self,
        single_action=SingleAction,
        multi_action=MultiAction,
    ):
        super().__init__(single_action, multi_action)

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
                assert len(join_key) != 0
                all_actions = all_actions.join(new_action, join_key)
        return all_actions
