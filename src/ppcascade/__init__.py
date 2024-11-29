from cascade.backends import register as register_backend
from earthkit.data  import SimpleFieldList

from .backends.fieldlist import SimpleFieldListBackend

register_backend(SimpleFieldList, SimpleFieldListBackend)
