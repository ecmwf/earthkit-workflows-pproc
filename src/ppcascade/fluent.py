import numpy as np
import xarray as xr
import itertools

from earthkit.data import FieldList
from cascade.fluent import Action, Node, Payload
from cascade.fluent import SingleAction as BaseSingleAction
from cascade.fluent import MultiAction as BaseMultiAction
from cascade.fluent import Fluent


from .utils.window import Range
from .utils.request import Request, MultiSourceRequest


class SingleAction(BaseSingleAction):
    def to_multi(self, nodes: xr.DataArray):
        return MultiAction(self, nodes, self.backend)

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
        assert len(windows) == 1
        assert self.nodes.coords[dim] == windows[0].name
        # Join with climatology and compute efi control
        payload = Payload(
            self.backend.efi,
            (Node.input_name(1), Node.input_name(0), eps),
            {"control": True},
        )
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

        def _sot(action: Action, number: int) -> Action:
            new_sot = action.reduce(
                Payload(
                    self.backend.sot,
                    (Node.input_name(1), Node.input_name(0), number, eps),
                )
            )
            new_sot._add_dimension(new_dim, number)
            return new_sot

        ret = self.join(climatology, "**datatype**", match_coord_values=True).transform(
            _sot, sot, new_dim
        )
        ret.non_descript_dim(new_dim)
        return ret

    def cluster(self, config, ncomp_file, indexes, deterministic):
        return self.map(
            Payload(
                self.backend.cluster,
                (config, Node.input_name(0), ncomp_file, indexes, deterministic),
            )
        )

    def write(self, target, config_grib_sets: dict):
        if target != "null:":
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
            payload = Payload(
                self.backend.write, (target, Node.input_name(0), grib_sets)
            )
            self.sinks.append(Node(payload, self.node()))
        return self


class MultiAction(BaseMultiAction):
    def to_single(self, payload_or_node: Payload | Node):
        if isinstance(payload_or_node, Payload):
            return SingleAction.from_payload(self, payload_or_node, self.backend)
        return SingleAction(self, payload_or_node, self.backend)

    def non_descript_dim(self, dim: str):
        """
        Mark in node attributes a dimension in node array that is not
        a metadata key
        """
        self.nodes.attrs.setdefault("grib_exclude", set())
        self.nodes.attrs["grib_exclude"].add(dim)

    def non_descript_dims(self):
        return self.nodes.attrs.pop("grib_exclude", [])

    def diff(self, dim: str = "", **method_kwargs):
        return self.reduce(
            Payload(self.backend.diff, kwargs=method_kwargs),
            dim,
        )

    def extreme(
        self,
        climatology: Action,
        windows: list[Range],
        sot: list[int],
        eps: float,
        efi_control: bool = False,
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
        efi_control: bool, whether to compute EFI for control member
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
        if efi_control:
            control = self.select({ensemble_dim: 0}, drop=True).efi(
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

        def _efi(action: Action, window: Range) -> Action:
            ret = action.select({dim: window.name}).reduce(
                Payload(
                    self.backend.efi,
                    (Node.input_name(1), Node.input_name(0), eps),
                ),
                dim="**datatype**",
            )
            return ret

        return self.join(
            climatology, "**datatype**", match_coord_values=True
        ).transform(_efi, windows, dim)

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

        def _sot_windows(action: Action, window: Range) -> Action:
            def _sot(action: Action, number: int) -> Action:
                new_sot = action.reduce(
                    Payload(
                        self.backend.sot,
                        (
                            Node.input_name(1),
                            Node.input_name(0),
                            number,
                            eps,
                        ),
                    )
                )
                new_sot._add_dimension(new_dim, number)
                return new_sot

            return action.select({dim: window.name}).transform(
                _sot, list(map(int, sot)), new_dim
            )

        ret = self.join(climatology, "**datatype**", match_coord_values=True).transform(
            _sot_windows, windows, dim
        )
        ret.non_descript_dim(new_dim)
        return ret

    def ensms(
        self,
        dim: str = "number",
        new_dim: str | xr.DataArray = xr.DataArray(["mean", "std"], name="type"),
    ):
        """
        Creates nodes computing the mean and standard deviation along the specified dimension. A new dimension
        is created joining the mean and standard deviation

        Parameters
        ----------
        dim: str, dimension to compute mean and standard deviation along
        new_dim: str or xr.DataArray, name of new dimension or xr.DataArray specifying new dimension name and
        coordinate values

        Return
        ------
        MultiAction
        """
        mean = self.mean(dim)
        std = self.std(dim)
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
    ):
        payload = Payload(
            self.backend.threshold,
            (
                Node.input_name(0),
                comparison,
                float(value),
                float(local_scale_factor) if local_scale_factor is not None else None,
                edition,
            ),
        )
        return (
            self.map(payload)
            .map(Payload(lambda x: FieldList.from_numpy(x.values * 100, x.metadata())))
            .mean(dim)
        )

    def anomaly(
        self,
        clim_mean: Action,
        clim_std: Action,
        std_anomaly: bool = False,
        **method_kwargs,
    ):
        anom = self.join(clim_mean, "**datatype**", match_coord_values=True).subtract(
            **method_kwargs
        )
        if not std_anomaly:
            return anom
        return anom.join(clim_std, "**datatype**", match_coord_values=True).divide()

    def quantiles(
        self, num_quantiles: int = 100, dim: str = "number", new_dim: str = "quantile"
    ):
        def _quantiles(action, quantile):
            payload = Payload(self.backend.quantiles, (Node.input_name(0), quantile))
            new_quantile = action.map(payload)
            new_quantile._add_dimension(new_dim, quantile)
            return new_quantile

        ret = self.concatenate(dim).transform(
            _quantiles, np.linspace(0.0, 1.0, int(num_quantiles) + 1), new_dim
        )
        ret.non_descript_dim(new_dim)
        return ret

    def wind_speed(self, vod2uv: bool, dim: str = "param"):
        if vod2uv:
            ret = self.map(Payload(self.backend.norm, (Node.input_name(0),)))
        else:
            ret = self.reduce(Payload(self.backend.norm), dim)
        return ret

    def param_operation(
        self, operation: str | Payload | None, dim: str = "param", **kwargs
    ):
        if operation is None:
            return self
        if isinstance(operation, str):
            if hasattr(self, operation):
                return getattr(self, operation)(dim=dim, **kwargs)
            operation = Payload(getattr(self.backend, operation), kwargs=kwargs)
        return self.reduce(operation, dim)

    def ensemble_operation(
        self, operation: str | Payload | None, dim: str = "number", **kwargs
    ):
        if operation is None:
            return self
        if isinstance(operation, str):
            if hasattr(self, operation):
                return getattr(self, operation)(dim=dim, **kwargs)
            operation = Payload(getattr(self.backend, operation), kwargs=kwargs)
        return self.reduce(operation, dim)

    def window_operation(self, operation: str, ranges: list[Range], dim: str = "step"):
        if operation is None:
            self._squeeze_dimension(dim)
            return self

        def _window_operation(action: Action, range: Range) -> Action:
            window_action = action.select({dim: range.steps}).reduce(
                Payload(
                    self.backend.window_operation,
                    kwargs={"operation": operation, "window": range},
                ),
                dim,
            )
            window_action._add_dimension(dim, range.name)
            return window_action

        return self.transform(_window_operation, ranges, dim)

    def pca(self, config, mask, target: str = None):
        if mask is not None:
            raise NotImplementedError()
        return self.reduce(
            Payload(
                self.backend.pca,
                (config, Node.input_name(0), Node.input_name(1), mask, target),
            )
        )

    def attribution(self, config, targets):
        def _attribution(action, scenario):
            payload = Payload(
                self.backend.attribution,
                (config, scenario, Node.input_name(0), Node.input_name(1)),
            )
            attr = action.reduce(payload)
            attr._add_dimension("scenario", scenario)
            return attr

        return self.transform(
            _attribution, ["centroids", "representatives"], "scenario"
        ).map(
            np.asarray(
                [
                    Payload(
                        self.backend.cluster_write,
                        (config, "centroids", Node.input_name(0), targets["centroids"]),
                    ),
                    Payload(
                        self.backend.cluster_write,
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
                    Payload(
                        self.backend.write, (target, Node.input_name(0), grib_sets)
                    ),
                    node,
                )
            )
        return self


from .backends.fieldlist import NumpyFieldListBackend


class PProcFluent(Fluent):
    def __init__(
        self,
        single_action=SingleAction,
        multi_action=MultiAction,
        backend=NumpyFieldListBackend,
    ):
        super().__init__(single_action, multi_action, backend)

    def source(
        self,
        requests: list[Request | MultiSourceRequest],
        join_key: str = "number",
        **kwargs,
    ):
        all_actions = None
        for request in requests:
            payloads = np.empty(tuple(request.dims.values()), dtype=object)
            for indices, new_request in request.expand():
                payloads[indices] = (new_request,)
            new_action = super().source(
                self.backend.retrieve,
                xr.DataArray(
                    payloads,
                    coords={key: list(request[key]) for key in request.dims.keys()},
                ),
                kwargs,
            )

            if all_actions is None:
                all_actions = new_action
            else:
                assert len(join_key) != 0
                all_actions = all_actions.join(new_action, join_key)
        return all_actions
