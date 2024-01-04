import numpy as np
import xarray as xr

from cascade.graph import Graph, deduplicate_nodes
from cascade.fluent import Payload

from .fluent import PProcFluent
from .io import retrieve
from .utils.request import Request, MultiSourceRequest
from .config import (
    Config,
    ClusterConfig,
)


def _read(
    requests: list[Request | MultiSourceRequest], join_key: str = "number", **kwargs
):
    all_actions = None
    for request in requests:
        payloads = np.empty(tuple(request.dims.values()), dtype=object)
        for indices, new_request in request.expand():
            payloads[indices] = Payload(
                retrieve,
                (new_request,),
                kwargs,
            )
        new_action = PProcFluent().source(
            payloads,
            dims={key: list(request[key]) for key in request.dims.keys()},
            name="retrieve",
            append_unique_index=False,
        )

        if all_actions is None:
            all_actions = new_action
        else:
            assert len(join_key) != 0
            all_actions = all_actions.join(new_action, join_key)
    return all_actions


def ensemble_anomaly(args):
    config = Config(args)
    total_graph = Graph([])
    for param_config in config.parameters:
        climatology = _read(
            param_config.clim_request(args.climatology), "step"
        )  # Will contain type em and es

        total_graph += (
            _read(param_config.forecast_request(args.forecast))
            .param_operation(
                param_config.param["operation"], **param_config.param["kwargs"]
            )
            .anomaly(
                climatology.select({"type": "em"}),
                climatology.select({"type": "es"}),
                param_config.windows.options.get("std_anomaly", False),
            )
            .window_operation(param_config.windows)
            .ensemble_operation(
                param_config.ensemble["operation"], **param_config.ensemble["kwargs"]
            )
            .write(
                param_config.target,
                param_config.out_keys,
            )
            .graph()
        )

    return deduplicate_nodes(total_graph)


def ensemble(args):
    config = Config(args)
    total_graph = Graph([])
    for param_config in config.parameters:
        requests = param_config.forecast_request(args.forecast)
        stream = False
        if isinstance(requests, tuple):
            requests, stream = requests
        total_graph += (
            _read(requests, stream=stream)
            .param_operation(
                param_config.param["operation"], **param_config.param["kwargs"]
            )
            .window_operation(param_config.windows)
            .ensemble_operation(
                param_config.ensemble["operation"], **param_config.ensemble["kwargs"]
            )
            .write(
                param_config.target,
                param_config.out_keys,
            )
            .graph()
        )
    return deduplicate_nodes(total_graph)


def extreme(args):
    config = Config(args)
    total_graph = Graph([])
    for param_config in config.parameters:
        climatology = _read(
            param_config.clim_request(args.climatology, True, no_expand=("quantile"))
        )
        total_graph += (
            _read(param_config.forecast_request(args.forecast))
            .param_operation(
                param_config.param["operation"], **param_config.param["kwargs"]
            )
            .window_operation(param_config.windows)
            .ensemble_extreme(
                param_config.ensemble["operation"],
                climatology,
                param_config.windows,
                **param_config.ensemble["kwargs"]
            )
            .write(
                param_config.target,
                param_config.out_keys,
            )
            .graph()
        )

    return deduplicate_nodes(total_graph)


def clustereps(args):
    config = ClusterConfig(args)
    if args.spread is not None:
        spread = _read(config.spread_request(args.spread, no_expand=("step",)))
    else:
        spread = _read(
            config.spread_compute_request(args.spread_compute, no_expand=("step",)),
            join_key="date",
        ).mean(dim="date")

    pca = (
        _read(config.forecast_request(args.ensemble, no_expand="step"))
        .concatenate(dim="number")
        .join(spread, dim="data_type")
        .pca(config, args.mask, args.pca)
    )
    cluster = pca.cluster(config, args.ncomp_file, args.indexes, args.deterministic)
    total_graph = (
        pca.join(cluster, dim="data_type")
        .attribution(
            config,
            {
                "centroids": (args.centroids, args.cen_anomalies),
                "representatives": (args.representative, args.rep_anomalies),
            },
        )
        .graph()
    )
    return deduplicate_nodes(total_graph)


GRAPHS = [ensemble_anomaly, ensemble, extreme, clustereps]
