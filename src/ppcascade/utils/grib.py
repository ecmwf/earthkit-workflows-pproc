from earthkit.data import FieldList

from .window import Range


def time_range_indicator(step: int) -> int:
    if step == 0:
        return 1
    if step > 255:
        return 10
    return 0


def window_grib_headers(operation: str, range: Range) -> dict:
    ret = {}
    try:
        ret["step"] = int(range.name)
        ret["timeRangeIndicator"] = time_range_indicator(ret["step"])
    except ValueError:
        ret["stepRange"] = range.name
        if operation == "diff":
            ret.update({"stepType": "diff", "timeRangeIndicator": 5})
        if operation == "mean":
            ret["timeRangeIndicator"] = 3
            ret["numberIncludedInAverage"] = len(range.steps)
            ret["numberMissingFromAveragesOrAccumulations"] = 0
        if operation in ["minimum", "maxmimum"]:
            ret["timeRangeIndicator"] = 2
        ret.setdefault("stepType", "max")
    return ret


def extreme_grib_headers(clim: FieldList, ens: FieldList) -> dict:
    extreme_headers = {}
    ens_template = ens.metadata()[0].buffer_to_metadata()

    # set clim keys
    clim_template = clim.metadata()[0].buffer_to_metadata()
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
        extreme_headers[key] = clim_template.get(key)

    fc_keys = [
        "date",
        "subCentre",
        "totalNumber",
    ]
    for key in fc_keys:
        extreme_headers[key] = ens_template.get(key)

    extreme_headers["ensembleSize"] = len(ens)

    return extreme_headers


def threshold_grib_headers(
    comparison: str, threshold: float, local_scale_factor: int, edition: int
) -> dict:
    if edition == 2:
        return {}

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
