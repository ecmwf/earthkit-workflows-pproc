import os

import yaml
from pproc.clustereps.__main__ import get_parser as cluster_parser
from pproc.common.config import default_parser


def basic_parser(description: str, clim: bool = False):
    parser = default_parser(description)
    parser.add_argument("-f", "--forecast", required=True, help="Forecast data (GRIB)")
    if clim:
        parser.add_argument(
            "--climatology", required=True, help="Climatology data (GRIB)"
        )
    return parser


_parsers = {
    "clustereps": cluster_parser(),
    "ensemble_anomaly": basic_parser(
        "Deterministic and ensemble forecast processing of anomalies", clim=True
    ),
    "ensemble": basic_parser(
        "Deterministic and ensemble forecast processing, including ensms, prob and quantiles"
    ),
    "extreme": basic_parser(
        "Compute EFI and SOT from forecast and climatology", clim=True
    ),
}


def get_parser(product: str):
    return _parsers.get(product)


class ArgsFile:
    def __init__(self, filename: str):
        with open(filename, "r") as f:
            self.defaults = yaml.safe_load(f)

    def args(self, product: str, config_file: str):
        args = ["-c", config_file]
        for key, value in self.defaults[product].items():
            if value == "none":
                continue
            if value == "REQUIRED":
                raise ValueError(
                    f"Required value not specified in default options for {key}!"
                )
            if value == "CWD":
                value = os.getcwd()
            args.extend([f"--{key}", f"{value}"])

        parser = get_parser(product)
        return parser.parse_args(args)
