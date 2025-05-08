# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import copy
import itertools
from collections import OrderedDict

import numpy as np


class Request:
    def __init__(self, request: dict, no_expand: tuple[str] = ()):
        self.request = request.copy()
        self.fake_dims = []
        self.no_expand = no_expand
        self.ignore = ["interpolate"]

    @property
    def dims(self) -> OrderedDict:
        dimensions = OrderedDict()
        for key, values in self.request.items():
            if key in self.ignore or key in self.no_expand:
                continue
            if hasattr(values, "__iter__") and not isinstance(values, str):
                dimensions[key] = len(values)
        return dimensions

    def __setitem__(self, key, value):
        self.request[key] = value

    def __getitem__(self, key):
        return self.request[key]

    def __contains__(self, key) -> bool:
        return key in self.request

    def update(self, **kwargs):
        self.request.update(**kwargs)

    def pop(self, key, default=None):
        if default is None:
            return self.request.pop(key)
        return self.request.pop(key, default)

    def make_dim(self, key, value=None):
        if key in self:
            assert type(self[key], (str, int, float))
            self[key] = [self[key]]
        else:
            self[key] = [value]
            self.fake_dims.append(key)

    def expand(self):
        reqs = [self.request[x] for x in self.dims.keys()]
        for indices, params in zip(
            itertools.product(*(range(len(x)) for x in reqs)), itertools.product(*reqs)
        ):
            new_request = self.request.copy()
            for index, expand_param in enumerate(self.dims.keys()):
                new_request[expand_param] = params[index]

            # Remove fake dims from request
            for dim in self.fake_dims:
                new_request.pop(dim)
            yield indices, new_request


class MultiSourceRequest(Request):
    def __init__(self, requests: list[dict], no_expand: tuple[str] = ()):
        super().__init__(requests[0], no_expand)
        self.requests = requests

    def __setitem__(self, key, value):
        super().__setattr__(key, value)
        [x.__setitem__(key, value) for x in self.requests]

    def __getitem__(self, key):
        values = [x.__getitem__(key) for x in self.requests]
        if np.all([values[0] == values[x] for x in range(1, len(values))]):
            return values[0]
        raise Exception(f"Requests {self.requests} differ on value for key {key}")

    def __contains__(self, key) -> bool:
        contains = [x.__contains__(key) for x in self.requests]
        if all([contains[0] == contains[x] for x in range(1, len(contains))]):
            return contains[0]
        raise Exception(f"Not all requests {self.requests} contain key {key}")

    def update(self, **kwargs):
        super().update(**kwargs)
        [x.update(**kwargs) for x in self.requests]

    def pop(self, key, default=None):
        contains = key in self
        if default is None or contains:
            value = self[key]
            super().pop(key)
            [x.pop(key) for x in self.requests]
            return value
        super().pop(key)
        [x.pop(key) for x in self.requests]
        return default

    def expand(self):
        for params in itertools.product(*[self.request[x] for x in self.dims.keys()]):
            indices = []
            new_requests = copy.deepcopy(self.requests)
            for index, expand_param in enumerate(self.dims.keys()):
                [x.__setitem__(expand_param, params[index]) for x in new_requests]
                indices.append(list(self.request[expand_param]).index(params[index]))

            # Remove fake dims from request
            for dim in self.fake_dims:
                [x.pop(dim) for x in new_requests]
            yield tuple(indices), new_requests
