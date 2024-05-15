from earthkit.data import FieldList
from earthkit.data.sources import array_list


class ArrayFieldList(array_list.ArrayFieldList):
    def __init__(self, array, metadata):
        super().__init__(array, metadata)

    def __getstate__(self) -> dict:
        return {
            "array": self.values,
            "metadata": self.metadata(),
        }

    def __setstate__(self, state: dict):
        new_fieldlist = FieldList.from_array(state["array"], state["metadata"])
        self.__dict__.update(new_fieldlist.__dict__)
        del new_fieldlist
