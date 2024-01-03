from cascade.fluent import Payload, Node
from ppcascade.entry import parser


def mock_args(config_path: str):
    test_parser = parser.basic_parser("Test", True, True)
    return test_parser.parse_args(
        ["--config", config_path, "--ensemble", "fdb:fc", "--climatology", "fdb:clim"]
    )


def mock_cluster_args(config_path: str):
    test_parser = parser.cluster_parser()
    return test_parser.parse_args(
        [
            "--config",
            config_path,
            "--ensemble",
            "fdb:ens_z500",
            "--deterministic",
            "fdb:determ_z500",
            "--date",
            "20231101",
            "--spread-compute",
            "fdb:spread_z500",
            "--spread-compute",
            "fileset:spread_z500",
            "--spread-compute",
            "mars:spread_z500",
            "--clim-dir",
            "",
        ]
    )


class MockPayload(Payload):
    def __init__(self, name: str):
        super().__init__(name)


class MockNode(Node):
    def __init__(self, name: str):
        super().__init__(MockPayload(name))
