import copy

from earthkit.data.readers.grib import metadata
from earthkit.data.readers.grib import memory
from earthkit.data.readers.grib import codes


class StandAloneGribMetadata(metadata.StandAloneGribMetadata):
    def __init__(self, handle, clear_data: bool = True):
        if clear_data:
            # Clear data values, keeping only metadata
            handle = handle.clone()
            handle.set_array("values", handle.get_array("values").shape)
        super().__init__(handle)

    def __getstate__(self) -> dict:
        ret = self.__dict__.copy()
        ret["_handle"] = self._handle.get_buffer()
        return ret

    def __setstate__(self, state: dict):
        state["_handle"] = codes.GribCodesHandle(
            memory.GribMessageMemoryReader(state["_handle"])._next_handle(), None, None
        )
        self.__dict__.update(state)

    def override(self, *args, **kwargs) -> "StandAloneGribMetadata":
        ret = super().override(*args, **kwargs)
        return StandAloneGribMetadata(ret._handle, clear_data=False)

    def _hide_internal_keys(self):
        return self
