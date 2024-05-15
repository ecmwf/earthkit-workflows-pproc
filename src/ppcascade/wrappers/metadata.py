import copy

from earthkit.data.readers.grib import metadata
from earthkit.data.readers.grib import memory
from earthkit.data.readers.grib import codes


class GribMetadata(metadata.GribMetadata):
    def __init__(self, handle, clear_data: bool = True):
        if clear_data:
            # Clear data values, keeping only metadata
            handle = handle.clone()
            handle.set_array("values", handle.get_array("values").shape)
        super().__init__(handle)

    def __getstate__(self) -> bytes:
        return self._handle.get_buffer()

    def __setstate__(self, state: bytes):
        self._handle = codes.GribCodesHandle(
            memory.GribMessageMemoryReader(state)._next_handle(), None, None
        )

    def override(self, *args, **kwargs) -> "GribMetadata":
        ret = super().override(*args, **kwargs)
        return GribMetadata(ret._handle, clear_data=False)

    def _hide_internal_keys(self):
        return self