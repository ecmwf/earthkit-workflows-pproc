# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import importlib
import inspect


class PatchModule:
    def __init__(self, module, original, patch):
        self.module = module
        self.original = importlib.import_module(original)
        self.patch = patch

    def __enter__(self):
        if self.original == self.patch:
            return

        functions = []
        module_key = None
        for key, value in self.module.__dict__.items():
            if value == self.original:
                module_key = key
            elif inspect.isfunction(value):
                functions.append(key)
        assert module_key is not None

        setattr(self.module, module_key, self.patch)
        for func in functions:
            func_source = inspect.getsource(getattr(self.module, func))
            exec(func_source, self.module.__dict__)

    def __exit__(self, type, value, traceback):
        if self.original == self.patch:
            return
        importlib.reload(self.module)
