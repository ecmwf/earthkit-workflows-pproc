import array_api_compat
import importlib
import os

from meteokit import extreme
from meteokit.stats import iter_quantiles
from cascade.backends.arrayapi import ArrayApiBackend as BaseArrayApiBackend
from cascade.backends.base import num_args

from ppcascade.utils.patch import PatchModule
from ppcascade.utils.window import Range
from ppcascade.utils.io import retrieve as ek_retrieve


def comp_str2func(array_module, comparison: str):
    if comparison == "<=":
        return array_module.less_equal
    if comparison == "<":
        return array_module.less
    if comparison == ">=":
        return array_module.greater_equal
    return array_module.greater


class ArrayAPIBackend(BaseArrayApiBackend):

    @num_args(2)
    def diff(*arrays):
        return arrays[1] - arrays[0]

    @num_args(2)
    def norm(*arrays):
        xp = array_api_compat.array_namespace(*arrays)
        return xp.sqrt(arrays[0] ** 2 + arrays[1] ** 2)

    def window_operation(*arrays, operation: str, window: Range):
        return getattr(ArrayAPIBackend, operation)(*arrays)

    def threshold(
        arr,
        comparison: str,
        value: float,
    ):
        xp = array_api_compat.array_namespace(arr)
        # Find all locations where nan appears as an ensemble value
        is_nan = xp.isnan(arr)
        thesh = comp_str2func(xp, comparison)(arr, value)
        return xp.where(is_nan, xp.nan, thesh)

    def efi(
        clim,
        ens,
        eps: float,
    ):
        xp = array_api_compat.array_namespace(ens, clim)
        with PatchModule(extreme, "numpy", xp):
            return extreme.efi(clim, ens, eps)

    def sot(
        clim,
        ens,
        number: int,
        eps: float,
    ):
        xp = array_api_compat.array_namespace(ens, clim)
        with PatchModule(extreme, "numpy", xp):
            return extreme.sot(clim, ens, number, eps)

    def quantiles(ens, quantile: float):
        xp = array_api_compat.array_namespace(ens)
        with PatchModule(extreme, "numpy", xp):
            return list(iter_quantiles(ens, [quantile], method="numpy"))[0]

    def filter(
        arr1,
        arr2,
        comparison: str,
        threshold: float,
        *,
        replacement: float = 0,
    ):
        xp = array_api_compat.array_namespace(arr1, arr2)
        condition = comp_str2func(xp, comparison)(arr2, threshold)
        return xp.where(condition, replacement, arr1)

    def pca(
        config,
        ens,
        spread,
        mask,
        target: str,
    ):
        raise NotImplementedError()

    def cluster(
        config,
        pca_output,
        ncomp_file: str,
        indexes: str,
        deterministic: str,
    ):
        raise NotImplementedError()

    def attribution(config, scenario: str, pca_output: dict, cluster_output: dict):
        raise NotImplementedError()

    def cluster_write(
        config,
        scenario,
        attribution_output,
        cluster_dests,
    ):
        raise NotImplementedError()

    def retrieve(request: dict | list[dict], **kwargs):
        xp = importlib.import_module(os.getenv("CASCADE_ARRAY_MODULE", "numpy"))
        res = ek_retrieve(request, **kwargs)
        return xp.asarray(res.values)

    def write(loc: str, data, metadata: dict):
        if loc != "null:":
            raise ValueError("Only null target supported for array api backend")
        # TODO: Currently just moves the data to CPU
        return array_api_compat.to_device(data, "cpu")
