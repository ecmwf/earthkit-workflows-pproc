import pytest
import os

from ppcascade import products

from helpers.mock import mock_args, mock_cluster_args

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


def test_cluster_graph():
    # With spread compute
    mock_args = mock_cluster_args(f"{ROOT_DIR}/templates/clustereps.yaml")
    graph = products.clustereps(mock_args)
    assert len([x for x in graph.nodes()]) == 64

    # With spread
    mock_args.spread = "fileset:spread_z500"
    graph = products.clustereps(mock_args)
    assert len([x for x in graph.nodes()]) == 33
