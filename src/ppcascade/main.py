import sys
import os
import argparse
import functools
import dill

from meters import ResourceMeter
from cascade.cascade import Cascade
from cascade.executors.dask import DaskLocalExecutor, DaskClientExecutor
from cascade.schedulers.depthfirst import DepthFirstScheduler
from cascade.contextgraph import ContextGraph
from cascade.graph import Graph, deduplicate_nodes, pyvis
from cascade.graph.export import serialise, deserialise
from cascade.transformers import to_task_graph

from ppcascade.entry.genconfig import RequestTranslator
from ppcascade.entry.parser import ArgsFile


def graph_from_serialisation(args):
    with open(args.graph_file, "rb") as f:
        data = dill.load(f)
    return deserialise(data)


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
        graph += Cascade.graph(config.graph_product, product_args, deduplicate=False)

    with ResourceMeter("MERGE"):
        deduplicate_nodes(graph)
    return graph


def graph_from_config(args):
    options = ArgsFile(args.options)
    product_args = options.args(args.product, args.config)
    return Cascade.graph(args.product, product_args)


def parse_options(options: str) -> dict:
    options = {}
    for pair in options.split(","):
        key, value = pair.split("=")
        try:
            value = int(value)
        except ValueError:
            pass
        options[key] = value
    return options


def benchmark(graph: Graph, output_root: str, b_options: str) -> dict:
    if not b_options:
        return {}
    options = parse_options(b_options)
    with ResourceMeter("BENCHMARK"):
        if b_options["type"] == "local":
            resource_map = DaskLocalExecutor().benchmark(
                graph,
                n_workers=options["workers"],
                memory_limit=f"{options['memory']}MB",
                adaptive=False,
                report=f"{output_root}/dask_report.html",
                mem_report=f"{output_root}/mem_usage.csv",
            )
        elif b_options["type"] == "client":
            resource_map = DaskClientExecutor().benchmark(
                graph,
                scheduler_file=options["scheduler_file"],
                adaptive=False,
                report=f"{output_root}/dask_report.html",
                mem_report=options["mem_report"],
            )
        else:
            raise ValueError(
                f"Unknown benchmark type {b_options['type']}. Expected one of local, client"
            )
    dill.dump(resource_map, open(f"{output_root}/resource_map.dill", "wb"))
    return resource_map


def schedule(graph: Graph, s_options: str, resource_map: dict) -> Graph:
    if not s_options:
        return graph
    options = parse_options(s_options)
    if not resource_map and options.get("resource_map", ""):
        resource_map = dill.load(open(options["resource_map"], "rb"))

    with ResourceMeter("SCHEDULE"):
        context_graph = ContextGraph()
        for index in range(options["workers"]):
            context_graph.add_node(
                f"worker-{index}",
                type="CPU",
                speed=1,
                memory=options["memory"],
            )
        for index in range(options["workers"] - 1):
            context_graph.add_edge(
                f"worker-{index}", f"worker-{index+1}", bandwidth=0.1, latency=1
            )
        graph = DepthFirstScheduler().schedule(
            to_task_graph(graph, resource_map), context_graph
        )
    return graph


def execute(graph: Graph, output_root: str, e_options: str):
    if not e_options:
        return
    options = parse_options(e_options)
    os.environ["DASK_LOGGING__DISTRIBUTED"] = "debug"
    with ResourceMeter("EXECUTE"):
        if options["type"] == "local":
            DaskLocalExecutor().execute(
                graph,
                n_workers=options["workers"],
                threads_per_worker=options["threads_per_worker"],
                memory_limit=f"{options['memory']}MB",
                adaptive=bool(options.get("adaptive", False)),
                report=f"{output_root}/dask_report.html",
            )
        elif options["type"] == "client":
            DaskClientExecutor().execute(
                graph,
                options["scheduler_file"],
                bool(options.get("adaptive", False)),
                report=f"{output_root}/dask_report.html",
            )
        else:
            raise ValueError(
                f"Unknown executor type {options['type']}. Expected one of local, client"
            )


def main(sys_args):
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description="Create and execute task graph for PPROC products"
    )
    parser.add_argument(
        "-o",
        "--output-root",
        type=str,
        default=os.getcwd(),
        help="Output directory for generated graph plot and execution report. Default: CWD",
    )
    parser.add_argument(
        "-p",
        "--plot",
        type=str,
        default="",
        help="Final name to plot graph to. If not provided, graph will not be plotting.",
    )
    parser.add_argument(
        "-s",
        "--serialise",
        type=str,
        default="",
        help="Filename to write serialised graph to. Graph is serialised using dill."
        + "If not provided, graph will not be serialised.",
    )

    dask_group = parser.add_argument_group("dask", "Dask execution options")
    dask_group.add_argument(
        "--benchmark-options",
        type=str,
        default="",
        help="Benchmarking options. If not provided, graph will not be benchmarked."
        + "Example: type=client,scheduler_file=/path/to/scheduler_file,mem_report=/path/to/mem_report.csv",
    )
    dask_group.add_argument(
        "--schedule-options",
        type=str,
        default="",
        help="Options for schedule graph using DepthFirstScheduler, memory in MB."
        + "If not provided, graph will not be scheduled. Example: workers=2,memory=10",
    )
    dask_group.add_argument(
        "--execute-options",
        type=str,
        default="",
        help="Execute graph using either local or client Dask executor, memory in MB."
        + "If not provided, graph will not be executed."
        + "Example: type=client,scheduler_file=/path/to/scheduler_file,adaptive=False",
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
    request_parser.add_argument(
        "--options",
        type=str,
        required=True,
        help="Path to default run options e.g. source specification",
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
    config_parser.add_argument(
        "--options",
        type=str,
        required=True,
        help="Path to default run options e.g. source specification",
    )
    config_parser.set_defaults(func=graph_from_config)

    graph_parser = subparsers.add_parser(
        "from_serialised", help="Load graph from dill serialisation file"
    )
    graph_parser.add_argument(
        "--file", type=str, required=True, help="Filename of serialised graph"
    )
    graph_parser.set_defaults(func=graph_from_serialisation)

    args = parser.parse_args(sys_args)
    graph = args.func(args)
    if len(args.plot) != 0:
        pyvis_graph = pyvis.to_pyvis(
            graph,
            notebook=True,
            cdn_resources="remote",
            height="1500px",
            node_attrs=functools.partial(node_info_ext, graph.sinks),
            hierarchical_layout=False,
        )
        pyvis_graph.show(args.plot)

    if len(args.serialise) != 0:
        data = serialise(graph)
        with open(args.serialise, "wb") as f:
            dill.dump(data, f)

    resource_map = benchmark(graph, args.output_root, args.benchmark_options)
    graph = schedule(graph, args.schedule_options, resource_map)
    execute(graph, args.output_root, args.execute_options)


if __name__ == "__main__":
    main(sys.argv[1:])
