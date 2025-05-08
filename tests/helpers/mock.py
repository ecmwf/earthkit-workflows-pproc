# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

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
