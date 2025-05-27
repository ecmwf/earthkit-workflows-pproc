# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
import pytest
import xarray as xr
from earthkit.workflows.fluent import Node, Payload

from earthkit.workflows import Graph, deduplicate_nodes
from earthkit.workflows.plugins.pproc.fluent import Action
import tempfile


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
    from earthkit.workflows.visualise import visualise

    with tempfile.TemporaryDirectory() as td:
        # NOTE `visualise` calls some dubious library which always generates javascript
        # in the current directory, despite the generated html being elsewhere. We chdir
        # to mitigate the pollution
        cwd = os.getcwd()
        os.chdir(td)
        visualise(graph, "test.html")
        os.chdir(cwd)
    nodes = list(graph.nodes())
    assert len(nodes) == nnodes
