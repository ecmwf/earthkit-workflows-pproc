import os

import pytest
from helpers.mock import mock_args

from ppcascade import products

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))


@pytest.mark.parametrize(
    "product, config, expected_num_nodes",
    [
        ["ensemble", f"{ROOT_DIR}/templates/prob.yaml", 119],
        ["ensemble_anomaly", f"{ROOT_DIR}/templates/t850.yaml", 93],
        ["ensemble", f"{ROOT_DIR}/templates/wind.yaml", 77],
        ["ensemble", f"{ROOT_DIR}/templates/ensms.yaml", 118],
        ["extreme", f"{ROOT_DIR}/templates/extreme.yaml", 81],
        ["ensemble", f"{ROOT_DIR}/templates/quantiles.yaml", 70],
    ],
)
def test_graph_construction(product, config, expected_num_nodes):
    graph = getattr(products, product)(mock_args(config))
    assert len([x for x in graph.nodes()]) == expected_num_nodes

