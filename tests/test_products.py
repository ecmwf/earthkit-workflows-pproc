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

from earthkit.workflows.plugins.pproc import products

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))


@pytest.mark.parametrize(
    "product, requests, expected_num_nodes",
    [
        ["ensemble", f"{ROOT_DIR}/templates/prob.yaml", 96],
        ["ensemble_anomaly", f"{ROOT_DIR}/templates/t850.yaml", 80],
        ["ensemble", f"{ROOT_DIR}/templates/ensms.yaml", 82],
        ["extreme", f"{ROOT_DIR}/templates/extreme.yaml", 80],
        ["ensemble", f"{ROOT_DIR}/templates/quantiles.yaml", 73],
    ],
    ids=["prob", "t850", "ensms", "extreme", "quantiles"],
)
def test_graph_construction(product, requests, expected_num_nodes):
    with open(requests, "r") as f:
        output_requests = yaml.safe_load(f)
    graph = getattr(products, product)(output_requests, f"{ROOT_DIR}/schema.yaml")
    assert len([x for x in graph.nodes()]) == expected_num_nodes
