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


def main(sys_args):
    sys.stdout.reconfigure(line_buffering=True)

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
    parser.add_argument("--serialise", default="", type=str, help="Filename to write serialised graph to. Graph is serialised using dill. If not provided, graph will not be serialised.")

    dask_group = parser.add_argument_group("dask", "Dask execution options")
    dask_group.add_argument(
        "--schedule", default=False, action="store_true", help="Schedule graph using DepthFirstScheduler"
    )
    dask_group.add_argument(
        "--execute", default=None, type=str, choices=["local", "client"], 
        help="Execute graph using either local or client Dask executor. If not provided, graph will not be executed. Default: None"
    )
    dask_group.add_argument(
        "--params", default="workers=2,threads_per_worker=1,memory=10", type=str, help="Comma separated list of parameters for Dask executor. Memory in GB. Default: workers=2,threads_per_worker=1,memory=10"
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

    graph_parser = subparsers.add_parser(
        "from_serialised", help="Load graph from dill serialisation file"
    )
    graph_parser.add_argument(
        "--graph-file", type=str, required=True, help="Filename of serialised graph"
    )
    graph_parser.set_defaults(func=graph_from_serialisation)

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

    if len(args.serialise) != 0:
        data = serialise(graph)
        with open(args.serialise, "wb") as f:
            dill.dump(data, f)

    if args.execute:
        executor_params = {}
        for pair in args.params.split(","):
            key, value = pair.split("=")
            try:
                value = int(value)
            except ValueError:
                pass
            executor_params[key] = value

        if args.schedule:
            with ResourceMeter("SCHEDULE"):
                context_graph = ContextGraph()
                for index in range(executor_params["workers"]):
                    context_graph.add_node(f"worker-{index}", type="CPU", speed=10, memory=executor_params["memory"])
                for index in range(executor_params["workers"] - 1):
                    context_graph.add_edge(f"worker-{index}", f"worker-{index+1}", bandwidth=0.1, latency=1)
                graph = DepthFirstScheduler().schedule(to_task_graph(graph, {}), context_graph)


        os.environ["DASK_LOGGING__DISTRIBUTED"] = "debug"
        with ResourceMeter("EXECUTE"):
            if args.execute == "local":
                DaskLocalExecutor().execute(
                    graph,
                    n_workers=executor_params["workers"],
                    threads_per_worker=executor_params["threads_per_worker"],
                    memory_limit=f"{executor_params['memory']}GB",
                    adaptive=bool(executor_params.get("adaptive", False)),
                    report=f"{args.output_root}/dask_report.html",
                )
            else:
                DaskClientExecutor().execute(
                    graph,
                    executor_params["scheduler_file"],
                    bool(executor_params.get("adaptive", False)),
                    report=f"{args.output_root}/dask_report.html",
                )
        
if __name__ == "__main__":
    main(sys.argv[1:])
