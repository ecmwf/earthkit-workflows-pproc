import sys
import os
import argparse

from meters import ResourceMeter
from cascade.cascade import Cascade
from cascade.executors.executor import Executor
from cascade.executors.dask import DaskLocalExecutor, DaskClientExecutor

from ppcascade.entry.genconfig import RequestTranslator
from ppcascade.entry.parser import ArgsFile
from ppcascade import products


def graph_from_request(args) -> Cascade:
    options = ArgsFile(args.options)
    translator = RequestTranslator(args.config)
    prods = translator.translate(args.request)
    cas = Cascade()
    for product, config in prods.items():
        config_file = f"{args.output_root}/{product}.yaml"
        config.to_yaml(config_file)
        product_args = options.args(config.graph_product, config_file)
        cas += Cascade(
            getattr(products, config.graph_product)(product_args, deduplicate=False)
        )
    return cas


def graph_from_config(args) -> Cascade:
    options = ArgsFile(args.options)
    product_args = options.args(args.product, args.config)
    return Cascade(getattr(products, args.product)(product_args))


def graph_from_serialisation(args) -> Cascade:
    return Cascade.from_serialised(args.file)


def parse_options(options_str: str) -> dict:
    options = {}
    if len(options_str) == 0:
        return options

    for pair in options_str.split(","):
        key, value = pair.split("=")
        options[key] = value
    return options


def create_executor(options: dict) -> Executor:
    if options["type"] == "local":
        return DaskLocalExecutor(
            n_workers=int(options["workers"]),
            threads_per_worker=int(options.get("threads_per_worker", 1)),
            memory_limit=f"{options['memory']}MB",
            adaptive_kwargs=parse_options(options.get("adaptive", "")),
        )

    elif options["type"] == "client":
        return DaskClientExecutor(
            dask_scheduler_file=options["scheduler_file"],
            adaptive=bool(options.get("adaptive", False)),
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
        "--benchmark",
        action="store_true",
        default=False,
        help="Benchmark tasks in graph. Default: False",
    )
    dask_group.add_argument(
        "--schedule",
        action="store_true",
        default=False,
        help="Schedule graph using DepthFirstScheduler. Default: False",
    )
    dask_group.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Execute graph using configured executor. Default: False",
    )
    dask_group.add_argument(
        "--executor-options",
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

    os.makedirs(args.output_root, exist_ok=True)
    cas = args.func(args)
    if len(args.plot) != 0:
        cas.visualise(args.plot)
    if len(args.serialise) != 0:
        cas.serialise(args.serialise)

    options = parse_options(args.executor_options)
    if len(options) > 0:
        cas.executor = create_executor(options)
        if args.benchmark:
            with ResourceMeter("BENCHMARK"):
                cas.benchmark(f"{args.output_root}/profiling")
            if len(args.serialise) != 0:
                cas.serialise(args.serialise)
        if args.schedule:
            with ResourceMeter("SCHEDULE"):
                cas.schedule()
        if args.execute:
            os.environ["DASK_LOGGING__DISTRIBUTED"] = "debug"
            with ResourceMeter("EXECUTE"):
                cas.execute()


if __name__ == "__main__":
    main(sys.argv[1:])
