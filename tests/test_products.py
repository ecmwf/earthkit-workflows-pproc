import pytest
import os

from cascade.cascade import Cascade

from helpers.mock import mock_args, mock_cluster_args

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))


@pytest.mark.parametrize(
    "product, config, expected_num_nodes",
    [
        ["ensemble", f"{ROOT_DIR}/templates/prob.yaml", 450],
        ["ensemble_anomaly", f"{ROOT_DIR}/templates/t850.yaml", 296],
        ["ensemble", f"{ROOT_DIR}/templates/wind.yaml", 168],
        ["ensemble", f"{ROOT_DIR}/templates/ensms.yaml", 180],
        ["extreme", f"{ROOT_DIR}/templates/extreme.yaml", 162],
        ["ensemble", f"{ROOT_DIR}/templates/quantiles.yaml", 92],
    ],
)
def test_graph_construction(product, config, expected_num_nodes):
    graph = Cascade.graph(product, mock_args(config))
    assert len([x for x in graph.nodes()]) == expected_num_nodes


def test_cluster_graph():
    # With spread compute
    mock_args = mock_cluster_args(f"{ROOT_DIR}/templates/clustereps.yaml")
    graph = Cascade.graph("clustereps", mock_args)
    assert len([x for x in graph.nodes()]) == 64

    # With spread
    mock_args.spread = "fileset:spread_z500"
    graph = Cascade.graph("clustereps", mock_args)
    assert len([x for x in graph.nodes()]) == 33


def test_unregistered():
    with pytest.raises(Exception):
        Cascade.graph("test")
