from cascade.graph import Graph, deduplicate_nodes

from pproc.config.utils import squeeze, expand

from ppcascade.fluent import from_source
from ppcascade.templates import derive_template
from ppcascade.utils.request import Request


def ensemble(
    output_requests: list[dict], pproc_schema: str, target: str = "null:"
) -> Graph:
    """
    Generate graph for ensemble member processing products e.g. ensms,
    quantiles, threshold probabilties.

    Parameters
    ----------
    output_requests: list[dict], list of dictionaries containing output requests
    pproc_schema: str, path to schema file
    target: str, target and location to write output

    Returns
    -------
    Graph
    """

    total_graph = Graph([])
    ensemble_dim = "type"
    for req in output_requests:
        config = derive_template(req, pproc_schema)
        source = from_source(
            [
                Request(x, no_expand=("number"))
                for x in squeeze(
                    sum([list(expand(x)) for x in config.inputs], []),
                    ["step", "number", "param", "levelist"],
                )
            ],
            join_key=ensemble_dim,
            backend_kwargs={"stream": True},
        ).concatenate(dim=ensemble_dim, keep_dim=True)
        total_graph += (
            config.action(forecast=source, ensemble_dim=ensemble_dim)
            .write(target)
            .graph()
        )
    return deduplicate_nodes(total_graph)


def ensemble_anomaly(
    output_requests: list[dict], pproc_schema: str, target: str = "null:"
) -> Graph:
    """
    Generate graph for ensemble member anomaly products e.g. threshold probabilities
    for t850.
    Parameters
    ----------
    output_requests: list[dict], list of dictionaries containing output requests
    pproc_schema: str, path to schema file
    target: str, target and location to write output

    Returns
    -------
    Graph
    """
    total_graph = Graph([])
    ensemble_dim = "type"
    for req in output_requests:
        config = derive_template(req, pproc_schema)
        forecast = from_source(
            [
                Request(x, no_expand=("number",))
                for x in squeeze(
                    sum([list(expand(x)) for x in config.forecast], []),
                    ["step", "number", "param", "levelist"],
                )
            ],
            join_key=ensemble_dim,
            backend_kwargs={"stream": True},
        ).concatenate(dim=ensemble_dim, keep_dim=True)
        climatology = from_source(
            [
                Request(x)
                for x in squeeze(
                    sum([list(expand(x)) for x in config.climatology], []),
                    ["type", "step"],
                )
            ],
            join_key="step",
        )
        total_graph += (
            config.action(
                forecast=forecast, climatology=climatology, ensemble_dim=ensemble_dim
            )
            .write(target)
            .graph()
        )
    return deduplicate_nodes(total_graph)


def extreme(
    output_requests: list[dict], pproc_schema: str, target: str = "null:"
) -> Graph:
    """
    Generate graph for EFI/SOT products.
    Parameters
    ----------
    output_requests: list[dict], list of dictionaries containing output requests
    pproc_schema: str, path to schema file
    target: str, target and location to write output

    Returns
    -------
    Graph
    """
    total_graph = Graph([])
    ensemble_dim = "type"
    for req in output_requests:
        config = derive_template(req, pproc_schema)
        forecast = from_source(
            [
                Request(x, no_expand=("number"))
                for x in squeeze(
                    sum([list(expand(x)) for x in config.forecast], []),
                    ["step", "number", "param", "levelist"],
                )
            ],
            join_key=ensemble_dim,
            backend_kwargs={"stream": True},
        ).concatenate(dim=ensemble_dim, keep_dim=True)
        climatology = from_source(
            [Request(x, no_expand=("quantile")) for x in config.climatology]
        )
        total_graph += (
            config.action(
                forecast=forecast, climatology=climatology, ensemble_dim=ensemble_dim
            )
            .write(target)
            .graph()
        )
    return deduplicate_nodes(total_graph)
