import argparse

from cascade.graph import Graph, deduplicate_nodes

from .fluent import PProcFluent
from .config import (
    Config,
    ClusterConfig,
)


def ensemble_anomaly(args: argparse.Namespace, deduplicate: bool = True):
    """
    Generate graph for ensemble member anomaly products e.g. threshold probabilities
    for t850. Input parameters args must contain the following attributes:
        forecast: str,  source for forecast data (GRIB)
        climatology: str, source for climatology data (GRIB)
        config: str, path to configuration file

    Parameters
    ----------
    args: argparse.Namespace containing attributes listed above
    deplicate: bool, whether to deduplicate nodes the graph, default is True

    Returns
    -------
    Graph
    """
    config = Config(args)
    total_graph = Graph([])
    fluent = PProcFluent()
    ensemble_dim = "type"
    for param_config in config.parameters:
        climatology = fluent.source(
            param_config.clim_request(args.climatology), join_key="step"
        )  # Will contain type em and es

        total_graph += (
            fluent.source(
                param_config.forecast_request(args.forecast, no_expand=("number",)),
                stream=True,
                join_key=ensemble_dim,
            )
            .concatenate(dim=ensemble_dim, keep_dim=True)
            .param_operation(
                param_config.param["operation"], **param_config.param["kwargs"]
            )
            .anomaly(
                climatology.select({"type": "em"}, drop=True),
                climatology.select({"type": "es"}, drop=True),
                param_config.windows.options.get("std_anomaly", False),
            )
            .window_operation(
                param_config.windows.operation,
                param_config.windows.ranges,
                batch_size=2,
            )
            .ensemble_operation(
                param_config.ensemble["operation"],
                dim=ensemble_dim,
                **param_config.ensemble["kwargs"],
            )
            .write(
                param_config.target,
                param_config.out_keys,
            )
            .graph()
        )

    if deduplicate:
        return deduplicate_nodes(total_graph)
    return total_graph


def ensemble(args: argparse.Namespace, deduplicate: bool = True):
    """
    Generate graph for ensemble member processing products e.g. ensms,
    quantiles, threshold probabilties. Input parameters
    args must contain the following attributes:
        forecast: str,  source for forecast data (GRIB)
        config: str, path to configuration file

    Parameters
    ----------
    args: argparse.Namespace containing attributes listed above
    deduplicate: bool, whether to deduplicate nodes the graph, default is True

    Returns
    -------
    Graph
    """
    config = Config(args)
    total_graph = Graph([])
    ensemble_dim = "type"
    for param_config in config.parameters:
        requests = param_config.forecast_request(args.forecast, no_expand=("number",))
        stream = True
        if isinstance(requests, tuple):
            requests, stream = requests
        total_graph += (
            PProcFluent()
            .source(requests, stream=stream, join_key=ensemble_dim)
            .concatenate(dim=ensemble_dim, keep_dim=True)
            .param_operation(
                param_config.param["operation"], **param_config.param["kwargs"]
            )
            .window_operation(
                param_config.windows.operation,
                param_config.windows.ranges,
                batch_size=2,
            )
            .ensemble_operation(
                param_config.ensemble["operation"],
                dim=ensemble_dim,
                **param_config.ensemble["kwargs"],
            )
            .write(
                param_config.target,
                param_config.out_keys,
            )
            .graph()
        )

    if deduplicate:
        return deduplicate_nodes(total_graph)
    return total_graph


def extreme(args: argparse.Namespace, deduplicate: bool = True):
    """
    Generate graph for EFI/SOT products. Input parameters
    args must contain the following attributes:
        forecast: str,  source for forecast data (GRIB)
        climatology: str, source for climatology data (GRIB)
        config: str, path to configuration file

    Parameters
    ----------
    args: argparse.Namespace containing attributes listed above
    deduplicate: bool, whether to deduplicate nodes the graph, default is True

    Returns
    -------
    Graph
    """
    config = Config(args)
    total_graph = Graph([])
    fluent = PProcFluent()
    for param_config in config.parameters:
        climatology = fluent.source(
            param_config.clim_request(args.climatology, True, no_expand=("quantile"))
        )
        ensemble_dim = "type"
        if "efi_control" in param_config.ensemble["kwargs"]:
            if param_config.ensemble["kwargs"]["efi_control"]:
                # If ensemble dim is "number" then criteria for the control should be {ensemble_dim: 0}
                param_config.ensemble["kwargs"]["efi_control"] = {ensemble_dim: "cf"}
            else:
                param_config.ensemble["kwargs"]["efi_control"] = None

        total_graph += (
            fluent.source(
                param_config.forecast_request(args.forecast, no_expand=("number",)),
                stream=True,
                join_key=ensemble_dim,
            )
            .param_operation(
                param_config.param["operation"], **param_config.param["kwargs"]
            )
            .window_operation(
                param_config.windows.operation,
                param_config.windows.ranges,
                batch_size=2,
            )
            .ensemble_extreme(
                param_config.ensemble["operation"],
                climatology,
                param_config.windows.ranges,
                ensemble_dim=ensemble_dim,
                **param_config.ensemble["kwargs"],
            )
            .write(
                param_config.target,
                param_config.out_keys,
            )
            .graph()
        )

    if deduplicate:
        return deduplicate_nodes(total_graph)
    return total_graph


def clustereps(args: argparse.Namespace, deduplicate: bool = True) -> Graph:
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
    deduplicate: bool, whether to deduplicate nodes the graph, default is True

    Returns
    -------
    Graph
    """
    config = ClusterConfig(args)
    fluent = PProcFluent()
    if args.spread is not None:
        spread = fluent.source(config.spread_request(args.spread, no_expand=("step",)))
    else:
        spread = fluent.source(
            config.spread_compute_request(args.spread_compute, no_expand=("step",)),
            join_key="date",
        ).mean(dim="date")

    pca = (
        fluent.source(
            config.forecast_request(args.ensemble, no_expand="step"), join_key="number"
        )
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

    if deduplicate:
        return deduplicate_nodes(total_graph)
    return total_graph


GRAPHS = [ensemble_anomaly, ensemble, extreme, clustereps]
