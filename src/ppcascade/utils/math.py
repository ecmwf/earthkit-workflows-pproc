import array_api_compat
import datetime
import copy
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
import thermofeel
from meters import ResourceMeter
from pproc.thermo.helpers import (
    compute_ehPa,
    field_values,
    find_utci_missing_values,
    get_datetime,
    latlon,
    step_interval,
    validate_utci,
)


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


def create_surface_output(values, template, **overrides):
    overrides = overrides.copy()
    overrides.update(
        {
            "typeOfFirstFixedSurface": 1,
            "scaleFactorOfFirstFixedSurface": "MISSING",
            "scaledValueOfFirstFixedSurface": "MISSING",
        }
    )
    return new_fieldlist(values, template, overrides)


def calc_cossza(*fields: FieldList, metadata: Metadata = None):
    all_fields = sum(fields[1:], fields[0])
    lats, lons = latlon(all_fields)
    _, validtime = get_datetime(all_fields)
    delta = step_interval(all_fields)
    dtbegin = validtime - datetime.timedelta(hours=delta)
    dtend = validtime
    cossza = earthkit.meteo.solar.cos_solar_zenith_angle_integrated(
        latitudes=lats,
        longitudes=lons,
        begin_date=dtbegin,
        end_date=dtend,
        integration_order=2,
    )
    t2m = field_values(all_fields, "2t")
    return new_fieldlist(
        cossza, t2m.metadata(), {**resolve_metadata(metadata), "paramId": "214001"}
    )


def calc_hmdx(*fields: FieldList, metadata: Metadata = None):
    all_fields = sum(fields[1:], fields[0])
    t2m = field_values(all_fields, "2t")
    t2d = field_values(all_fields, "2d")
    hmdx = thermofeel.calculate_humidex(t2_k=t2m, td_k=t2d)
    return create_surface_output(
        hmdx, t2m.metadata(), {**resolve_metadata(metadata), "paramId": "261016"}
    )


def calc_rhp(*fields: FieldList, metadata: Metadata = None):
    all_fields = sum(fields[1:], fields[0])
    t2m = field_values(all_fields, "2t")
    t2d = field_values(all_fields, "2d")
    rhp = thermofeel.calculate_relative_humidity_percent(t2_k=t2m, td_k=t2d)
    return new_fieldlist(
        rhp, t2m.metdata(), {**resolve_metadata(metadata), "paramId": "260242"}
    )


def calc_heatx(*fields: FieldList, metadata: Metadata = None):
    all_fields = sum(fields[1:], fields[0])
    t2m = field_values(all_fields, "2t")
    t2d = field_values(all_fields, "2d")
    heatx = thermofeel.calculate_heat_index_adjusted(t2_k=t2m, td_k=t2d)
    return create_surface_output(
        heatx, t2m.metadata(), {**resolve_metadata(metadata), "paramId": "260004"}
    )


def calc_dsrp(*fields: FieldList, metadata: Metadata = None):
    """
    In the absence of dsrp, approximate it with fdir and cossza.
    Note this introduces some amount of error as cossza approaches zero
    """
    all_fields = sum(fields[1:], fields[0])
    if "dsrp" in all_fields.indices()["param"]:
        return all_fields.sel(param="dsrp")

    fdir = field_values(all_fields, "fdir")
    cossza = field_values(all_fields, "cossza")
    dsrp = thermofeel.approximate_dsrp(fdir, cossza)
    return new_fieldlist(
        dsrp, fdir.metadata(), {**resolve_metadata(metadata), "paramId": "47"}
    )


def calc_utci(*fields: FieldList, metadata: Metadata = None, validate=True):
    all_fields = sum(fields[1:], fields[0])
    t2m = field_values(all_fields, "2t")
    t2d = field_values(all_fields, "2d")
    ws = field_values(all_fields, "10si")
    mrt = field_values(all_fields, "mrt")

    lats, lons = latlon(all_fields)
    ehPa = compute_ehPa(t2m, t2d)
    utci = thermofeel.calculate_utci(t2_k=t2m, va=ws, mrt=mrt, ehPa=ehPa)
    xp = array_api_compat.array_namespace(utci.values)
    for index in range(len(utci)):
        missing = find_utci_missing_values(
            t2m[index],
            ws[index],
            mrt[index],
            ehPa[index],
            utci[index],
            False,
        )
        if validate:
            validate_utci(utci[index], missing, lats, lons)
        utci[index][missing] = xp.nan
    return create_surface_output(
        utci, t2m.metadata(), {**resolve_metadata(metadata), "paramId": "261001"}
    )


def calc_wbgt(*fields: FieldList, metadata: Metadata = None):
    all_fields = sum(fields[1:], fields[0])
    t2m = field_values(all_fields, "2t")
    t2d = field_values(all_fields, "2d")
    ws = field_values(all_fields, "10si")
    mrt = field_values(all_fields, "mrt")

    wbgt = thermofeel.calculate_wbgt(t2m, mrt, ws, t2d)
    return create_surface_output(
        wbgt, t2m.metadata(), {**resolve_metadata(metadata), "paramId": "261014"}
    )


def calc_gt(*fields: FieldList, metadata: Metadata = None):
    all_fields = sum(fields[1:], fields[0])
    t2m = field_values(all_fields, "2t")
    ws = field_values(all_fields, "10si")
    mrt = field_values(all_fields, "mrt")

    gt = thermofeel.calculate_bgt(t2m, mrt, ws)
    return create_surface_output(
        gt, t2m.metadata(), {**resolve_metadata(metadata), "paramId": "261015"}
    )


def calc_nefft(*fields: FieldList, metadata: Metadata = None):
    all_fields = sum(fields[1:], fields[0])
    t2m = field_values(all_fields, "2t")
    rhp = field_values(all_fields, "rhp")
    ws = field_values(all_fields, "10si")

    nefft = thermofeel.calculate_normal_effective_temperature(t2m, ws, rhp)
    return create_surface_output(
        nefft, t2m.metadata(), {**resolve_metadata(metadata), "paramId": "261018"}
    )


def calc_wcf(*fields: FieldList, metadata: Metadata = None):
    all_fields = sum(fields[1:], fields[0])
    t2m = field_values(all_fields, "2t")
    ws = field_values(all_fields, "10si")

    wcf = thermofeel.calculate_wind_chill(t2m, ws)  # Kelvin
    return create_surface_output(
        wcf, t2m.metadata(), {**resolve_metadata(metadata), "paramId": "260005"}
    )


def calc_aptmp(*fields: FieldList, metadata: Metadata = None):
    all_fields = sum(fields[1:], fields[0])
    t2m = field_values(all_fields, "2t")
    rhp = field_values(all_fields, "rhp")
    ws = field_values(all_fields, "10si")

    aptmp = thermofeel.calculate_apparent_temperature(t2_k=t2m, va=ws, rh=rhp)
    return create_surface_output(
        aptmp, t2m.metadata(), {**resolve_metadata(metadata), "paramId": "260255"}
    )


def calc_mrt(*fields: FieldList, metadata: Metadata = None):
    all_fields = sum(fields[1:], fields[0])
    ssrd = field_values(all_fields, "ssrd")
    fdir = field_values(all_fields, "fdir")
    strd = field_values(all_fields, "strd")
    ssr = field_values(all_fields, "ssr")
    dsrp = field_values(all_fields, "dsrp")
    strr = field_values(all_fields, "strr")
    cossza = field_values(all_fields, "cossza")

    delta = step_interval(all_fields)
    seconds_in_time_step = delta * 3600  # steps are in hours

    f = 1.0 / float(seconds_in_time_step)
    # remove negative values from deaccumulated solar fields
    for v in ssrd, fdir, strd, ssr:
        v[v < 0] = 0
    mrt = thermofeel.calculate_mean_radiant_temperature(
        ssrd * f, ssr * f, dsrp * f, strd * f, fdir * f, strr * f, cossza
    )
    return create_surface_output(
        mrt, cossza.metadata(), {**resolve_metadata(metadata), "paramId": "261002"}
    )
