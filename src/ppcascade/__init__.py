from cascade.cascade import register_graph
from cascade.backends import register as register_backend
from earthkit.data.sources.numpy_list import NumpyFieldList

from .backends.arrayapi import ArrayAPIBackend
from .backends.fieldlist import NumpyFieldListBackend
from .products import GRAPHS

for product in GRAPHS:
    register_graph(product.__name__, product)

register_backend("default", ArrayAPIBackend)
register_backend(NumpyFieldList, NumpyFieldListBackend)
