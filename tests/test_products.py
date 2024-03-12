import pytest
import os
import functools
from cascade.graph import pyvis

from ppcascade.main import node_info_ext

from cascade.cascade import Cascade

from helpers.mock import mock_args, mock_cluster_args

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))


@pytest.mark.parametrize(
    "product, config, expected_num_nodes",
    [
        ["ensemble", f"{ROOT_DIR}/templates/prob.yaml", 112],
        ["ensemble_anomaly", f"{ROOT_DIR}/templates/t850.yaml", 120],
        ["ensemble", f"{ROOT_DIR}/templates/wind.yaml", 72],
        ["ensemble", f"{ROOT_DIR}/templates/ensms.yaml", 100],
        ["extreme", f"{ROOT_DIR}/templates/extreme.yaml", 74],
        ["ensemble", f"{ROOT_DIR}/templates/quantiles.yaml", 60],
    ],
)
def test_graph_construction(product, config, expected_num_nodes):
    graph = Cascade.graph(product, mock_args(config))
    pyvis_graph = pyvis.to_pyvis(
            graph,
            notebook=True,
            cdn_resources="remote",
            height="1500px",
            node_attrs=functools.partial(node_info_ext, graph.sinks),
            hierarchical_layout=False,
        )
    pyvis_graph.show(f"{product}_graph.html")
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
