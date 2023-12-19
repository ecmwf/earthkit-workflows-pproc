import numpy as np
import xarray as xr
import itertools

from earthkit.data import FieldList
from cascade.fluent import Action, Node, Payload
from cascade.fluent import SingleAction as BaseSingleAction
from cascade.fluent import MultiAction as BaseMultiAction
from cascade.fluent import Fluent


from .io import cluster_write
from .io import write as write_grib
from .graph_config import (
    Window,
)
from .fieldlist_backend import NumpyFieldListBackend


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

    def efi(self, climatology: Action, eps: float, num_steps: int):
        # Join with climatology and compute efi control
        payload = Payload(
            NumpyFieldListBackend.efi,
            (Node.input_name(1), Node.input_name(0), eps, num_steps),
            {"control": True},
        )
        return self.join(climatology, "**datatype**").reduce(payload)

    def cluster(self, config, ncomp_file, indexes, deterministic):
        return self.map(
            Payload(
                NumpyFieldListBackend.cluster,
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
            payload = Payload(write_grib, (target, Node.input_name(0), grib_sets))
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

    def diff(self, dim: str = "", **method_kwargs):
        return self.reduce(
            Payload(NumpyFieldListBackend.diff, kwargs=method_kwargs),
            dim,
        )

    def extreme(
        self,
        climatology: Action,
        sot: list,
        eps: float,
        num_steps: int,
        efi_control: bool = False,
        dim: str = "number",
        new_dim: str = "type",
    ):
        efi = self.efi(climatology, eps, num_steps, dim)
        efi._add_dimension(new_dim, "efi")
        if efi_control:
            control = self.select({dim: 0}).efi(climatology, eps, num_steps)
            control._add_dimension(new_dim, "efic")
            efi = efi.join(control, new_dim)
        sot = self.sot(climatology, eps, sot, num_steps, dim, new_dim)
        ret = efi.join(sot, new_dim)
        ret.non_descript_dim(new_dim)
        return ret

    def efi(self, climatology: Action, eps: float, num_steps: int, dim: str = "number"):
        return (
            self.concatenate(dim)
            .join(climatology, "**datatype**")
            .reduce(
                Payload(
                    NumpyFieldListBackend.efi,
                    (Node.input_name(1), Node.input_name(0), eps, num_steps),
                )
            )
        )

    def sot(
        self,
        climatology: Action,
        eps: float,
        sot: list[int],
        num_steps: int,
        dim: str = "number",
        new_dim: str = "sot",
    ):
        def _sot(action: Action, number: int):
            new_sot = action.reduce(
                Payload(
                    NumpyFieldListBackend.sot,
                    (Node.input_name(1), Node.input_name(0), number, eps, num_steps),
                )
            )
            new_sot._add_dimension(new_dim, number)
            return new_sot

        ret = (
            self.concatenate(dim)
            .join(climatology, "**datatype**")
            .transform(_sot, sot, new_dim)
        )
        ret.non_descript_dim(new_dim)
        return ret

    def ensms(
        self,
        dim: str = "number",
        new_dim: str | xr.DataArray = xr.DataArray(["mean", "std"], name="type"),
    ):
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
        dim: str = "number",
    ):
        payload = Payload(
            NumpyFieldListBackend.threshold,
            (
                Node.input_name(0),
                comparison,
                value,
                local_scale_factor,
            ),  # Needs edition!!
        )
        return (
            self.map(payload)
            .map(Payload(lambda x: FieldList.from_numpy(x.values * 100, x.metadata())))
            .mean(dim)
        )

    def anomaly(self, climatology: Action, window: Window):
        extract = (
            ("climateDateFrom", "climateDateTo", "referenceDate")
            if window.grib_set.get("edition", 1) == 2
            else ()
        )

        anom = self.join(
            climatology.select({"type": "em"}), "**datatype**", match_coord_values=True
        ).subtract(extract_keys=extract)

        if window.options.get("std_anomaly", False):
            anom = anom.join(
                climatology.select({"type": "es"}),
                "**datatype**",
                match_coord_values=True,
            ).divide()
        return anom

    def quantiles(
        self, num_quantiles: int = 100, dim: str = "number", new_dim: str = "quantile"
    ):
        def _quantiles(action, quantile):
            payload = Payload(
                NumpyFieldListBackend.quantiles, (Node.input_name(0), quantile)
            )
            if isinstance(action, BaseSingleAction):
                new_quantile = action.map(payload)
            else:
                new_quantile = action.map(payload)
            new_quantile._add_dimension(new_dim, quantile)
            return new_quantile

        ret = self.concatenate(dim).transform(
            _quantiles, np.linspace(0.0, 1.0, num_quantiles + 1), new_dim
        )
        ret.non_descript_dim(new_dim)
        return ret

    def wind_speed(self, vod2uv: bool, dim: str = "param"):
        if vod2uv:
            ret = self.map(Payload(NumpyFieldListBackend.norm, (Node.input_name(0),)))
        else:
            ret = self.reduce(Payload(NumpyFieldListBackend.norm), dim)
        return ret

    def param_operation(
        self, operation: str | Payload | None, dim: str = "param", **kwargs
    ):
        if operation is None:
            return self
        if isinstance(operation, str):
            if hasattr(self, operation):
                return getattr(self, operation)(dim=dim, **kwargs)
            operation = Payload(
                getattr(NumpyFieldListBackend, operation), kwargs=kwargs
            )
        return self.reduce(operation, dim)

    def ensemble_operation(
        self, operation: str | Payload | None, dim: str = "number", **kwargs
    ):
        if operation is None:
            return self
        if isinstance(operation, str):
            if hasattr(self, operation):
                return getattr(self, operation)(dim=dim, **kwargs)
            operation = Payload(
                getattr(NumpyFieldListBackend, operation), kwargs=kwargs
            )
        return self.reduce(operation, dim)

    def window_operation(self, window: Window, dim: str = "step"):
        if window.operation is None:
            self._squeeze_dimension(dim)
            ret = self
        else:
            ret = getattr(self, window.operation)(dim)
        if window.end - window.start == 0:
            ret.add_attributes({"step": window.name})
        else:
            ret.add_attributes({"stepRange": window.name})
        return ret

    def pca(self, config, mask, target: str = None):
        if mask is not None:
            raise NotImplementedError()
        return self.reduce(
            Payload(
                NumpyFieldListBackend.pca,
                (config, Node.input_name(0), Node.input_name(1), mask, target),
            )
        )

    def attribution(self, config, targets):
        def _attribution(action, scenario):
            payload = Payload(
                NumpyFieldListBackend.attribution,
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
                        cluster_write,
                        (config, "centroids", Node.input_name(0), targets["centroids"]),
                    ),
                    Payload(
                        cluster_write,
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
        if target != "null:":
            coords = list(self.nodes.coords.keys())
            exclude = self.non_descript_dims()
            for node_attrs in itertools.product(
                *[self.nodes.coords[key].data for key in coords]
            ):
                node_coords = {
                    key: node_attrs[index] for index, key in enumerate(coords)
                }
                node = self.node(node_coords)

                grib_sets = config_grib_sets.copy()
                grib_sets.update(self.nodes.attrs)
                grib_sets.update(
                    {k: v for k, v in node_coords.items() if k not in exclude}
                )
                self.sinks.append(
                    Node(
                        Payload(write_grib, (target, Node.input_name(0), grib_sets)),
                        [node],
                    )
                )
        return self


class PProcFluent(Fluent):
    single_action: type = SingleAction
    multi_action: type = MultiAction
