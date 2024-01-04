import sys
import os
import argparse
import functools

from cascade.cascade import Cascade
from cascade.contextgraph import ContextGraph
from cascade.scheduler import Schedule
from cascade.executor import DaskExecutor
from cascade.graph import Graph, deduplicate_nodes, pyvis

from ppcascade.entry.genconfig import RequestTranslator
from ppcascade.entry.parser import ArgsFile


def node_info_ext(sinks, node):
    info = pyvis.node_info(node)
    info["color"] = "#648FFF"
    if not node.inputs:
        info["shape"] = "diamond"
        info["color"] = "#DC267F"
    elif node in sinks:
        info["shape"] = "triangle"
        info["color"] = "#FFB000"
    if node.payload is not None:
        t = []
        if "title" in info:
            t.append(info["title"])
        func, *args = node.payload
        t.append(f"Function: {func}")
        if args:
            t.append("Arguments:")
            t.extend(f"- {arg!r}" for arg in args)
        info["title"] = "\n".join(t)
    return info


def graph_from_request(args):
    options = ArgsFile(args.options)
    translator = RequestTranslator(args.config)
    products = translator.translate(args.request)
    graph = Graph([])
    for product, config in products.items():
        config_file = f"{args.output_root}/{product}.yaml"
        config.to_yaml(config_file)
        product_args = options.args(config.graph_product, config_file)
        graph += Cascade.graph(config.graph_product, product_args)
    deduplicate_nodes(graph)
    return graph


def graph_from_config(args):
    options = ArgsFile(args.options)
    product_args = options.args(args.product, args.config)
    return Cascade.graph(args.product, product_args)


def main(sys_args):
    parser = argparse.ArgumentParser(
        description="Create and execute task graph for PPROC products"
    )
    parser.add_argument(
        "--options",
        type=str,
        required=True,
        help="Path to default run options e.g. source specification",
    )
    parser.add_argument(
        "--output-root",
        default=os.getcwd(),
        help="Output directory for generated graph plot and execution report. Default: CWD",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        default=False,
        help="Plot final graph. Default: False",
    )

    dask_group = parser.add_argument_group("dask", "Dask execution options")
    dask_group.add_argument(
        "--memory",
        default="10GB",
        type=str,
        help="Memory limit of Dask workers. Default: 10GB",
    )
    dask_group.add_argument(
        "--workers", default=2, type=int, help="Number of Dask workers. Default: 2"
    )
    dask_group.add_argument(
        "--threads-per-worker",
        default=1,
        type=int,
        help="Number of threads per Dask worker. Default: 1",
    )

    subparsers = parser.add_subparsers(required=True)
    request_parser = subparsers.add_parser(
        "from_request", help="Generate graph from MARS like request"
    )
    request_parser.add_argument(
        "-r", "--request", type=str, required=True, help="Path to request file"
    )
    request_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to default config for parameters",
    )
    request_parser.set_defaults(func=graph_from_request)

    config_parser = subparsers.add_parser(
        "from_config", help="Generate graph from graph configuration file"
    )
    config_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to graph configuration file",
    )
    config_parser.add_argument(
        "--product",
        type=str,
        required=True,
        choices=["ensemble", "ensemble_anomaly", "extreme", "clustereps"],
        help="Graph product associated to configuration file",
    )
    config_parser.set_defaults(func=graph_from_config)

    args = parser.parse_args(sys_args)
    graph = args.func(args)
    if args.plot:
        pyvis_graph = pyvis.to_pyvis(
            graph,
            notebook=True,
            cdn_resources="remote",
            height="1500px",
            node_attrs=functools.partial(node_info_ext, graph.sinks),
            hierarchical_layout=False,
        )
        pyvis_graph.show(f"{args.output_root}/graph.html")
    executor = DaskExecutor(Schedule(graph, ContextGraph(), {}))
    executor.execute(
        args.memory,
        args.workers,
        args.threads_per_worker,
        f"{args.output_root}/dask_report.html",
    )


if __name__ == "__main__":
    main(sys.argv[1:])
