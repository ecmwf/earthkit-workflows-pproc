import array_api_compat
from cascade.backends.earthkit import (
    Metadata,
    resolve_metadata,
    comp_str2func,
    new_fieldlist,
)
from earthkit.data import FieldList
from earthkit.meteo.extreme import array as extreme
from earthkit.meteo.stats import array as stats
import earthkit.meteo.solar
from meters import ResourceMeter
from pproc.thermo.indices import ComputeIndices


from ppcascade.utils import grib
from ppcascade.utils.patch import PatchModule


def threshold(
    arr: FieldList,
    comparison: str,
    value: float,
    *,
    metadata: Metadata = None,
) -> FieldList:
    with ResourceMeter("THRESHOLD"):
        xp = array_api_compat.array_namespace(arr.values)
        # Find all locations where nan appears as an ensemble value
        is_nan = xp.isnan(arr.values)
        thesh = comp_str2func(xp, comparison)(arr.values, value)
        res = xp.where(is_nan, xp.nan, thesh)
        return new_fieldlist(res, arr.metadata(), resolve_metadata(metadata, arr))


def efi(
    clim: FieldList,
    ens: FieldList,
    eps: float,
    *,
    metadata: Metadata = None,
) -> FieldList:
    with ResourceMeter(f"EFI, clim {clim.values.shape}, ens {ens.values.shape}"):
        xp = array_api_compat.array_namespace(ens.values, clim.values)
        with PatchModule(extreme, "numpy", xp):
            res = extreme.efi(clim.values, ens.values, eps)
        return new_fieldlist(
            res,
            [ens[0].metadata()],
            {**resolve_metadata(metadata, clim, ens), **grib.efi(clim, ens)},
        )


def sot(
    clim: FieldList,
    ens: FieldList,
    number: int,
    eps: float,
    *,
    metadata: Metadata = None,
) -> FieldList:
    with ResourceMeter("SOT"):
        xp = array_api_compat.array_namespace(ens.values, clim.values)
        with PatchModule(extreme, "numpy", xp):
            res = extreme.sot(clim.values, ens.values, number, eps)
        return new_fieldlist(
            res,
            [ens[0].metadata()],
            {
                **resolve_metadata(metadata, clim, ens),
                **grib.sot(clim, ens, number),
            },
        )


def quantiles(
    ens: FieldList, quantile: float, *, metadata: Metadata = None
) -> FieldList:
    with ResourceMeter("QUANTILES"):
        xp = array_api_compat.array_namespace(ens.values)
        with PatchModule(stats, "numpy", xp):
            res = list(stats.iter_quantiles(ens.values, [quantile], method="numpy"))[0]
        return new_fieldlist(
            res,
            [ens[0].metadata()],
            {**resolve_metadata(metadata, ens), "perturbationNumber": quantile},
        )


def _check_shape(fields: earthkit.data.FieldList, params: list[str]):
    shape = None
    for param in params:
        selected = fields.sel(param=param)
        if len(selected) == 0:
            raise ValueError(
                f"Field {param} not found in fields: \n {fields.ls(namespace='mars')}"
            )
        if shape is None:
            shape = selected.values.shape
        assert (
            shape == selected.values.shape
        ), f"Shape mismatch for {param} {shape} != {selected.values.shape}"


def calc_cossza(*fields: FieldList, metadata: Metadata = None):
    all_fields = sum(fields[1:], fields[0])
    return ComputeIndices(resolve_metadata(metadata)).calc_cossza_int(all_fields)


def calc_hmdx(*fields: FieldList, metadata: Metadata = None):
    all_fields = sum(fields[1:], fields[0])
    _check_shape(all_fields, ["2t", "2d"])
    return ComputeIndices(resolve_metadata(metadata)).calc_hmdx(all_fields)


def calc_rhp(*fields: FieldList, metadata: Metadata = None):
    all_fields = sum(fields[1:], fields[0])
    _check_shape(all_fields, ["2t", "2d"])
    return ComputeIndices(resolve_metadata(metadata)).calc_rhp(all_fields)


def calc_heatx(*fields: FieldList, metadata: Metadata = None):
    all_fields = sum(fields[1:], fields[0])
    _check_shape(all_fields, ["2t", "2d"])
    return ComputeIndices(resolve_metadata(metadata)).calc_heatx(all_fields)


def calc_dsrp(*fields: FieldList, metadata: Metadata = None):
    all_fields = sum(fields[1:], fields[0])
    return ComputeIndices(resolve_metadata(metadata)).calc_dsrp(all_fields)


def calc_utci(*fields: FieldList, metadata: Metadata = None, validate=True):
    all_fields = sum(fields[1:], fields[0])
    _check_shape(all_fields, ["2t", "2d", "10si", "mrt"])
    return ComputeIndices(resolve_metadata(metadata)).calc_utci(
        all_fields, print_misses=False, validate=validate
    )


def calc_wbgt(*fields: FieldList, metadata: Metadata = None):
    all_fields = sum(fields[1:], fields[0])
    _check_shape(all_fields, ["2t", "2d", "10si", "mrt"])
    return ComputeIndices(resolve_metadata(metadata)).calc_wbgt(all_fields)


def calc_gt(*fields: FieldList, metadata: Metadata = None):
    all_fields = sum(fields[1:], fields[0])
    _check_shape(all_fields, ["2t", "10si", "mrt"])
    return ComputeIndices(resolve_metadata(metadata)).calc_gt(all_fields)


def calc_nefft(*fields: FieldList, metadata: Metadata = None):
    all_fields = sum(fields[1:], fields[0])
    _check_shape(all_fields, ["2t", "10si", "2r"])
    return ComputeIndices(resolve_metadata(metadata)).calc_nefft(all_fields)


def calc_wcf(*fields: FieldList, metadata: Metadata = None):
    all_fields = sum(fields[1:], fields[0])
    _check_shape(all_fields, ["2t", "10si"])
    return ComputeIndices(resolve_metadata(metadata)).calc_wcf(all_fields)


def calc_aptmp(*fields: FieldList, metadata: Metadata = None):
    all_fields = sum(fields[1:], fields[0])
    _check_shape(all_fields, ["2t", "10si", "2r"])
    return ComputeIndices(resolve_metadata(metadata)).calc_aptmp(all_fields)


def calc_mrt(*fields: FieldList, metadata: Metadata = None):
    all_fields = sum(fields[1:], fields[0])
    _check_shape(all_fields, ["ssrd", "fdir", "strd", "ssr", "dsrp", "str", "uvcossza"])
    return ComputeIndices(resolve_metadata(metadata)).calc_mrt(all_fields)
