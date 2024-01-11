import numpy as np
import argparse

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


def ensemble_anomaly(args: argparse.Namespace):
    """
    Generate graph for ensemble member anomaly products e.g. threshold probabilities
    for t850. Input parameters args must contain the following attributes:
        forecast: str,  source for forecast data (GRIB)
        climatology: str, source for climatology data (GRIB)
        config: str, path to configuration file

    Parameters
    ----------
    args: argparse.Namespace containing attributes listed above

    Returns
    -------
    Graph
    """
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


def ensemble(args: argparse.Namespace):
    """
    Generate graph for ensemble member processing products e.g. ensms,
    quantiles, threshold probabilties. Input parameters
    args must contain the following attributes:
        forecast: str,  source for forecast data (GRIB)
        config: str, path to configuration file

    Parameters
    ----------
    args: argparse.Namespace containing attributes listed above

    Returns
    -------
    Graph
    """
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


def extreme(args: argparse.Namespace):
    """
    Generate graph for EFI/SOT products. Input parameters
    args must contain the following attributes:
        forecast: str,  source for forecast data (GRIB)
        climatology: str, source for climatology data (GRIB)
        config: str, path to configuration file

    Parameters
    ----------
    args: argparse.Namespace containing attributes listed above

    Returns
    -------
    Graph
    """
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


def clustereps(args: argparse.Namespace) -> Graph:
    """
    Generate graph for clustereps related products. Input parameters
    args must contain the following attributes:
        mask: str | None,
        spread: str | None, source for ensemble spread (GRIB)
        spread_compute: list[str], source for ensemble spread computation (GRIB)
        ensemble: str,  source for forecast data (GRIB)
        deterministic: str, source for deterministic forecast data (GRIB)
        clim_dir: str, climatology data root directory
        pca: str | None, PCA outputs (NPZ)
        centroids: str | None, cluster centroids output (GRIB)
        representative: str | None, representative members output (GRIB)
        cen_anomalies: str | None, cluster centroids output in anomaly space (GRIB)
        rep_anomalies: str | None, cluster representative members output in anomaly space (GRIB)
        indexes: str | None, cluster indexes output (NPZ)
        comp_file: str | None, number of components output (text)
        output_root: str, output directory for reports
        config: str, path to configuration file

    Parameters
    ----------
    args: argparse.Namespace containing attributes listed above

    Returns
    -------
    Graph
    """
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
