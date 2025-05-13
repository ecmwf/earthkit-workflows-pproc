from typing import Any

from earthkit.data import FieldList
from pproc.prob.parallel import threshold_grib_headers


def window(operation: str, coords: list[Any], include_init: bool) -> dict:
    if len(coords) == 1:
        return {}

    ret = {}
    if operation == "diff":
        ret.update({"timeRangeIndicator": 5, "stepType": "diff"})
    if operation == "mean":
        ret["timeRangeIndicator"] = 3
        ret["numberIncludedInAverage"] = (
            len(coords) if include_init else len(coords) - 1
        )
        ret["numberMissingFromAveragesOrAccumulations"] = 0
    if operation in ["min", "max"]:
        ret["timeRangeIndicator"] = 2
    ret.setdefault("stepType", "max")
    ret["stepRange"] = f"{coords[0]}-{coords[-1]}"
    return ret


def extreme(clim: FieldList, ens: FieldList, metadata: dict) -> dict:
    edition = metadata.get("edition", ens[0].metadata().get("edition", 1))
    clim_edition = clim[0].metadata().get("edition", 1)
    ret = {}
    if edition == 1 and clim_edition == 1:
        # set clim keys
        clim_keys = [
            "versionNumberOfExperimentalSuite",
            "implementationDateOfModelCycle",
            "numberOfReforecastYearsInModelClimate",
            "numberOfDaysInClimateSamplingWindow",
            "sampleSizeOfModelClimate",
            "versionOfModelClimate",
        ]
        for key in clim_keys:
            ret[key] = clim[0].metadata()[key]

        # set fc keys
        fc_keys = [
            "date",
            "subCentre",
            "totalNumber",
        ]
        for key in fc_keys:
            ret[key] = ens[0].metadata()[key]
    elif edition == 2 and clim_edition == 2:
        clim_keys = [
            "typeOfReferenceDataset",
            "yearOfStartOfReferencePeriod",
            "dayOfStartOfReferencePeriod",
            "monthOfStartOfReferencePeriod",
            "hourOfStartOfReferencePeriod",
            "minuteOfStartOfReferencePeriod",
            "secondOfStartOfReferencePeriod",
            "sampleSizeOfReferencePeriod",
            "numberOfReferencePeriodTimeRanges",
            "typeOfStatisticalProcessingForTimeRangeForReferencePeriod",
            "indicatorOfUnitForTimeRangeForReferencePeriod",
            "lengthOfTimeRangeForReferencePeriod",
        ]
        ret.update(
            {
                "productDefinitionTemplateNumber": 105,
                **{key: clim[0].metadata()[key] for key in clim_keys},
            }
        )
    else:
        raise Exception(
            f"Unsupported GRIB edition {edition} and clim edition {clim_edition}"
        )
    return ret


def efi(ens: FieldList, clim: FieldList, metadata: dict) -> dict:
    ret = extreme(ens, clim, metadata)
    edition = metadata.get("edition", ens[0].metadata().get("edition", 1))
    if edition not in [1, 2]:
        raise Exception(f"Unsupported GRIB edition {edition}")

    if len(ens) == 1 and ens.metadata()[0].get("type") in ["cf", "fc"]:
        ret["marsType"] = 28
        if edition == 1:
            ret["efiOrder"] = 0
            ret["totalNumber"] = 1
            ret["number"] = 0
        else:
            ret.update(
                {"typeOfRelationToReferenceDataset": 20, "typeOfProcessedData": 3}
            )
    else:
        ret["marsType"] = 27
        if edition == 1:
            ret["efiOrder"] = 0
            ret["number"] = 0
        else:
            ret.update(
                {"typeOfRelationToReferenceDataset": 20, "typeOfProcessedData": 5}
            )
    return ret


def sot(ens: FieldList, clim: FieldList, metadata: dict, number: int) -> dict:
    ret = extreme(ens, clim, metadata)
    ret["marsType"] = 38

    if number == 90:
        efi_order = 99
    elif number == 10:
        efi_order = 1
    else:
        raise Exception(
            f"SOT value '{number}' not supported in template! Only accepting 10 and 90"
        )
    edition = metadata.get("edition", ens[0].metadata().get("edition", 1))
    if edition == 1:
        ret["number"] = sot
        ret["efiOrder"] = efi_order
    elif edition == 2:
        ret.update(
            {
                "typeOfRelationToReferenceDataset": 21,
                "typeOfProcessedData": 5,
                "numberOfAdditionalParametersForReferencePeriod": 2,
                "scaleFactorOfAdditionalParameterForReferencePeriod": [0, 0],
                "scaledValueOfAdditionalParameterForReferencePeriod": [sot, efi_order],
            }
        )
    else:
        raise Exception(f"Unsupported GRIB edition {edition}")
    return ret


def threshold(
    edition: int, comparison: str, threshold: float, local_scale_factor: int | None
) -> dict:
    threshold_config = {
        "value": float(threshold),
        "comparison": comparison,
        "out_paramid": 0,
    }
    if local_scale_factor:
        threshold_config["local_scale_factor"] = local_scale_factor
    ret = threshold_grib_headers(
        edition,
        threshold_config,
    )
    ret.pop("paramId")
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


def quantiles(
    ens: FieldList, metadata: dict, pert_number: int, total_number: int
) -> dict:
    edition = metadata.get("edition", ens[0].metadata().get("edition", 1))
    if edition == 1:
        return {
            "totalNumber": total_number,
            "perturbationNumber": pert_number,
        }
    return {
        "productDefinitionTemplateNumber": metadata.get(
            "productDefinitionTemplateNumber", 86
        ),
        "totalNumberOfQuantiles": total_number,
        "quantileValue": pert_number,
    }
