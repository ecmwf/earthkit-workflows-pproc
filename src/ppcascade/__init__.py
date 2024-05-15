from cascade.backends import register as register_backend
from ppcascade.wrappers.array_list import ArrayFieldList

from .backends.fieldlist import ArrayFieldListBackend

register_backend(ArrayFieldList, ArrayFieldListBackend)
