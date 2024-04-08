import functools

from earthkit.data import FieldList

from .window import Range


def window(operation: str, range: Range) -> dict:
    # Note: don't need to step for len(range.steps) == 1, as should already
    # be correct in template
    if len(range.steps) == 1:
        return {}

    ret = {"stepRange": range.name}
    if operation == "diff":
        ret.update({"stepType": "diff", "timeRangeIndicator": 5})
    if operation == "mean":
        ret["timeRangeIndicator"] = 3
        ret["numberIncludedInAverage"] = len(range.steps)
        ret["numberMissingFromAveragesOrAccumulations"] = 0
    if operation in ["min", "max"]:
        ret["timeRangeIndicator"] = 2
    ret.setdefault("stepType", "max")
    return ret


def extreme(clim: FieldList, ens: FieldList) -> dict:
    extreme_headers = {}

    # set clim keys
    clim_keys = [
        "powerOfTenUsedToScaleClimateWeight",
        "weightAppliedToClimateMonth1",
        "firstMonthUsedToBuildClimateMonth1",
        "lastMonthUsedToBuildClimateMonth1",
        "firstMonthUsedToBuildClimateMonth2",
        "lastMonthUsedToBuildClimateMonth2",
        "numberOfBitsContainingEachPackedValue",
    ]
    for key in clim_keys:
        extreme_headers[key] = clim.metadata()[0].get(key)

    fc_keys = [
        "date",
        "subCentre",
        "totalNumber",
    ]
    for key in fc_keys:
        extreme_headers[key] = ens.metadata()[0].get(key)

    return extreme_headers


def efi(ens: FieldList, clim: FieldList) -> dict:
    ret = extreme(ens, clim)
    if len(ens) == 1 and ens.metadata()[0].get("type") == "cf":
        ret.update({"marsType": "efic", "totalNumber": 1, "number": 0})
    else:
        ret.update({"marsType": "efi", "efiOrder": 0, "number": 0})
    return ret


def sot(ens: FieldList, clim: FieldList, number: int) -> dict:
    ret = extreme(ens, clim)
    if number == 90:
        efi_order = 99
    elif number == 10:
        efi_order = 1
    else:
        raise Exception(
            "SOT value '{sot}' not supported in template! Only accepting 10 and 90"
        )
    ret.update(
        {
            "marsType": "sot",
            "efiOrder": efi_order,
            "number": number,
        }
    )
    return ret


def threshold(
    comparison: str, threshold: float, local_scale_factor: int | None
) -> dict:
    """
    Generated metadata required for computing threshold probabilities. Note,
    only required for GRIB edition 1 products
    """

    ret = {}
    if local_scale_factor is not None:
        ret["localDecimalScaleFactor"] = local_scale_factor
        threshold_value = round(threshold * 10**local_scale_factor, 0)
    else:
        threshold_value = threshold

    if "<" in comparison:
        ret.update({"thresholdIndicator": 2, "upperThreshold": threshold_value})
    elif ">" in comparison:
        ret.update({"thresholdIndicator": 1, "lowerThreshold": threshold_value})
    return ret


def anomaly_clim(clim: FieldList) -> dict:
    """
    Get required information from climatology metadata. Note,
    only required for GRIB edition 2 threshold probability products
    """
    return {
        key: clim.metadata()[0].get(key)
        for key in ["climateDateFrom", "climateDateTo", "referenceDate"]
    }
