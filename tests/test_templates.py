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
import yaml
from earthkit.workflows.graph import Graph, deduplicate_nodes
from pproc.config.utils import expand, squeeze

from earthkit.workflows.plugins.pproc.fluent import from_source
from earthkit.workflows.plugins.pproc.templates import derive_template, from_request
from earthkit.workflows.plugins.pproc.utils.request import Request

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

sources = from_source(
    [
        Request(
            {
                "param": "130",
                "levtype": "pl",
                "levelist": [250, 850],
                "type": "pf",
                "number": list(range(0, 5)),
            }
        ),
        Request(
            {
                "param": "130",
                "levtype": "pl",
                "levelist": [250, 850],
                "type": "cf",
            }
        ),
    ],
    join_key="type",
    backend_kwargs={"stream": True},
)


@pytest.mark.parametrize(
    "requests, expected_num_nodes",
    [
        [f"{ROOT_DIR}/templates/prob.yaml", 96],
        [f"{ROOT_DIR}/templates/ensms.yaml", 82],
        [f"{ROOT_DIR}/templates/quantiles.yaml", 73],
    ],
    ids=["prob", "ensms", "quantiles"],
)
def test_from_request(requests, expected_num_nodes):
    with open(requests, "r") as f:
        output_requests = yaml.safe_load(f)

    graph = Graph([])
    for req in output_requests:
        config = derive_template(req, f"{ROOT_DIR}/schema.yaml")
        source = from_source(
            [
                Request(x, no_expand=("number"))
                for x in squeeze(
                    sum([list(expand(x)) for x in config.inputs], []),
                    ["step", "number", "param", "levelist"],
                )
            ],
            join_key="type",
            backend_kwargs={"stream": True},
        ).concatenate(dim="type", keep_dim=True)
        graph += (
            from_request(
                req,
                f"{ROOT_DIR}/schema.yaml",
                ensemble_dim="type",
                forecast=source,
                metadata={"edition": 2},
            )
            .write("null:")
            .graph()
        )

    graph = deduplicate_nodes(graph)
    assert len([x for x in graph.nodes()]) == expected_num_nodes
