from cascade.backends import register as register_backend
from earthkit.data.sources.numpy_list import NumpyFieldList

from .backends.fieldlist import NumpyFieldListBackend

register_backend(NumpyFieldList, NumpyFieldListBackend)
