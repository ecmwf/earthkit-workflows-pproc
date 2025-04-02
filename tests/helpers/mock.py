from earthkit.workflows.fluent import Node, Payload

from ppcascade.entry import parser


def mock_args(config_path: str):
    test_parser = parser.basic_parser("Test", True)
    return test_parser.parse_args(
        ["--config", config_path, "--forecast", "fdb:fc", "--climatology", "fdb:clim"]
    )


class MockPayload(Payload):
    def __init__(self, name: str):
        super().__init__(name)


class MockNode(Node):
    def __init__(self, name: str):
        super().__init__(MockPayload(name))
