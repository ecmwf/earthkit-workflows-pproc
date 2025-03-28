import numpy as np
import xarray as xr
import pytest

from cascade.graph import Graph, deduplicate_nodes
from cascade.fluent import Node, Payload
from ppcascade.fluent import Action


@pytest.mark.parametrize(
    "inputs, nnodes",
    [
        [[["2t", "2d", "10u", "10v"], ["2t", "10u", "10v"]], 8],
        [[["167", "169", "165", "166"], ["167", "165", "166"]], 9],
    ],
    ids=["match-names", "no-match"],
)
def test_thermal(inputs, nnodes):
    action = Action(
        xr.DataArray(
            data=[
                Node(Payload(print, [x]), name=str(x)) for x in range(len(inputs[0]))
            ],
            dims=["param"],
            coords={"param": inputs[0]},
        )
    )
    graph = Graph([])
    for index, param in enumerate(["nefft", "wcf"]):
        new_action = action.sel(param=inputs[index]).thermal_index(param)
        graph += new_action.graph()
    graph = deduplicate_nodes(graph)
    nodes = list(graph.nodes())
    assert len(nodes) == nnodes
