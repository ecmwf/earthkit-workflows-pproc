import array_api_compat
import numpy as np
from meters import ResourceMeter
from os.path import join as pjoin

from meteokit import extreme
from meteokit.stats import iter_quantiles
from earthkit.data import FieldList
from earthkit.data.sources.numpy_list import NumpyFieldList
from pproc.common.io import (
    target_from_location,
    write_grib,
    FileTarget,
    FileSetTarget,
)
from pproc.common.resources import ResourceMeter
from pproc import clustereps
from pproc.clustereps.utils import normalise_angles
from pproc.clustereps.io import read_steps_grib
from pproc.clustereps.__main__ import write_cluster_attr_grib
from pproc.clustereps.cluster import get_output_keys
from cascade.backends import num_args

from ppcascade.utils.grib import (
    window_grib_headers,
    extreme_grib_headers,
    threshold_grib_headers,
)
from ppcascade.wrappers.metadata import GribBufferMetaData
from ppcascade.utils.patch import PatchModule
from ppcascade.utils.window import Range
from ppcascade.utils.io import retrieve as ek_retrieve


def standardise_output(data):
    # Also, nest the data to avoid problems with not finding geography attribute
    if len(data.shape) == 1:
        data = data.reshape((1, *data.shape))
    assert len(data.shape) == 2
    return data


def comp_str2func(array_module, comparison: str):
    if comparison == "<=":
        return array_module.less_equal
    if comparison == "<":
        return array_module.less
    if comparison == ">=":
        return array_module.greater_equal
    return array_module.greater


class NumpyFieldListBackend:
    def _merge(*fieldlists: list[NumpyFieldList]):
        """
        Merge fieldlist elements into a single array. fieldlists with
        different number of fields must be concatenated, otherwise, the
        elements in each fieldlist are stacked along a new dimension
        """
        if len(fieldlists) == 1:
            return fieldlists[0].values

        values = [x.values for x in fieldlists]
        xp = array_api_compat.array_namespace(*values)
        return xp.asarray(values)

    def multi_arg_function(func: str, *arrays: list[NumpyFieldList]) -> NumpyFieldList:
        with ResourceMeter(func.upper()):
            merged_array = NumpyFieldListBackend._merge(*arrays)
            xp = array_api_compat.array_namespace(*merged_array)
            res = standardise_output(getattr(xp, func)(merged_array, axis=0))
            return FieldList.from_numpy(
                res, [arrays[0][x].metadata() for x in range(len(res))]
            )

    def two_arg_function(
        func: str, *arrays: NumpyFieldList, extract_keys: tuple = ()
    ) -> NumpyFieldList:
        with ResourceMeter(func.upper()):
            # First argument must be FieldList
            assert isinstance(arrays[0], FieldList)
            val1 = arrays[0].values
            if isinstance(arrays[1], FieldList):
                val2 = arrays[1].values
                arr2_meta = arrays[1][0].metadata().buffer_to_metadata()
                override = {key: arr2_meta.get(key) for key in extract_keys}
                xp = array_api_compat.array_namespace(val1, val2)
            else:
                val2 = arrays[1]
                override = {}
                xp = array_api_compat.array_namespace(val1)

            res = getattr(xp, func)(val1, val2)
            metadata = [
                arrays[0][x].metadata().override(override) for x in range(len(res))
            ]
            return FieldList.from_numpy(standardise_output(res), metadata)

    def mean(*arrays: list[NumpyFieldList]) -> NumpyFieldList:
        return NumpyFieldListBackend.multi_arg_function("mean", *arrays)

    def std(*arrays: list[NumpyFieldList]) -> NumpyFieldList:
        return NumpyFieldListBackend.multi_arg_function("std", *arrays)

    def min(*arrays: list[NumpyFieldList]) -> NumpyFieldList:
        return NumpyFieldListBackend.multi_arg_function("min", *arrays)

    def max(*arrays: list[NumpyFieldList]) -> NumpyFieldList:
        return NumpyFieldListBackend.multi_arg_function("max", *arrays)

    def sum(*arrays: list[NumpyFieldList]) -> NumpyFieldList:
        return NumpyFieldListBackend.multi_arg_function("sum", *arrays)

    def prod(*arrays: list[NumpyFieldList]) -> NumpyFieldList:
        return NumpyFieldListBackend.multi_arg_function("prod", *arrays)

    def var(*arrays: list[NumpyFieldList]) -> NumpyFieldList:
        return NumpyFieldListBackend.multi_arg_function("var", *arrays)

    def stack(*arrays: list[NumpyFieldList], axis: int = 0) -> NumpyFieldList:
        if axis != 0:
            raise ValueError("Can not stack FieldList along axis != 0")
        assert np.all(
            [len(x) == 1 for x in arrays]
        ), "Can not stack FieldLists with more than one element, use concat"
        return NumpyFieldListBackend.concat(*arrays)

    def add(*arrays: list[NumpyFieldList], extract_keys: tuple = ()) -> NumpyFieldList:
        return NumpyFieldListBackend.two_arg_function(
            "add", *arrays, extract_keys=extract_keys
        )

    def subtract(
        *arrays: list[NumpyFieldList], extract_keys: tuple = ()
    ) -> NumpyFieldList:
        return NumpyFieldListBackend.two_arg_function(
            "subtract", *arrays, extract_keys=extract_keys
        )

    @num_args(2)
    def diff(*arrays: list[NumpyFieldList], extract_keys: tuple = ()) -> NumpyFieldList:
        return NumpyFieldListBackend.multiply(
            NumpyFieldListBackend.subtract(*arrays, extract_keys=extract_keys), -1
        )

    def multiply(
        *arrays: list[NumpyFieldList], extract_keys: tuple = ()
    ) -> NumpyFieldList:
        return NumpyFieldListBackend.two_arg_function(
            "multiply", *arrays, extract_keys=extract_keys
        )

    def divide(
        *arrays: list[NumpyFieldList], extract_keys: tuple = ()
    ) -> NumpyFieldList:
        return NumpyFieldListBackend.two_arg_function(
            "divide", *arrays, extract_keys=extract_keys
        )

    def pow(*arrays: list[NumpyFieldList], extract_keys: tuple = ()) -> NumpyFieldList:
        return NumpyFieldListBackend.two_arg_function(
            "pow", *arrays, extract_keys=extract_keys
        )

    def concat(*arrays: list[NumpyFieldList]) -> NumpyFieldList:
        """
        Concatenates the list of fields inside each NumpyFieldList into a single
        NumpyFieldList object

        Parameters
        ----------
        arrays: list[NumpyFieldList]
            NumpyFieldList instances to whose fields are to be concatenated

        Return
        ------
        NumpyFieldList
            Contains all fields inside the input field lists
        """
        return sum(arrays[1:], arrays[0])

    def take(array: NumpyFieldList, indices: int | tuple, *, axis: int):
        if axis != 0:
            raise ValueError("Can not take from FieldList along axis != 0")
        if isinstance(indices, int):
            indices = [indices]
        return array[indices]

    def norm(*arrays: list[NumpyFieldList]) -> NumpyFieldList:
        merged_array = NumpyFieldListBackend._merge(*arrays)
        xp = array_api_compat.array_namespace(merged_array)
        norm = standardise_output(xp.sqrt(xp.sum(xp.pow(merged_array, 2), axis=0)))
        return FieldList.from_numpy(
            norm, [arrays[0][x].metadata() for x in range(len(norm))]
        )

    def threshold(
        arr: NumpyFieldList,
        comparison: str,
        value: float,
        local_scale_factor=None,
        edition: int = 1,
    ) -> NumpyFieldList:
        with ResourceMeter("THRESHOLD"):
            xp = array_api_compat.array_namespace(arr.values)
            # Find all locations where np.nan appears as an ensemble value
            is_nan = xp.isnan(arr.values)
            thesh = comp_str2func(xp, comparison)(arr.values, value)
            res = xp.where(is_nan, xp.nan, thesh)
            threshold_headers = threshold_grib_headers(
                comparison, value, local_scale_factor, edition
            )
            metadata = [
                arr[x].metadata().override(threshold_headers) for x in range(len(res))
            ]
            return FieldList.from_numpy(standardise_output(res), metadata)

    def efi(
        clim: NumpyFieldList,
        ens: NumpyFieldList,
        eps: float,
    ) -> NumpyFieldList:
        with ResourceMeter(f"EFI, clim {clim.values.shape}, ens {ens.values.shape}"):
            extreme_headers = extreme_grib_headers(clim, ens)
            if False:
                # TODO: whether this is efi control should be deduced from grib data
                extreme_headers.update(
                    {"marsType": "efic", "totalNumber": 1, "number": 0}
                )
            else:
                extreme_headers.update({"marsType": "efi", "efiOrder": 0, "number": 0})
            metadata = ens[0].metadata().override(extreme_headers)
            xp = array_api_compat.array_namespace(ens.values, clim.values)
            with PatchModule(extreme, "numpy", xp):
                res = extreme.efi(clim.values, ens.values, eps)
            return FieldList.from_numpy(standardise_output(res), metadata)

    def sot(
        clim: NumpyFieldList,
        ens: NumpyFieldList,
        number: int,
        eps: float,
    ) -> NumpyFieldList:
        with ResourceMeter("SOT"):
            extreme_headers = extreme_grib_headers(clim, ens)
            if number == 90:
                efi_order = 99
            elif number == 10:
                efi_order = 1
            else:
                raise Exception(
                    "SOT value '{sot}' not supported in template! Only accepting 10 and 90"
                )
            metadata = (
                ens[0]
                .metadata()
                .override(
                    {
                        **extreme_headers,
                        "marsType": "sot",
                        "efiOrder": efi_order,
                        "number": number,
                    }
                )
            )

            xp = array_api_compat.array_namespace(ens.values, clim.values)
            with PatchModule(extreme, "numpy", xp):
                res = extreme.sot(clim.values, ens.values, number, eps)
            return FieldList.from_numpy(standardise_output(res), metadata)

    def quantiles(ens: NumpyFieldList, quantile: float) -> NumpyFieldList:
        with ResourceMeter("QUANTILES"):
            xp = array_api_compat.array_namespace(ens.values)
            with PatchModule(extreme, "numpy", xp):
                res = list(iter_quantiles(ens.values, [quantile], method="numpy"))[0]
            return FieldList.from_numpy(
                standardise_output(res),
                ens[0].metadata().override({"perturbationNumber": quantile}),
            )

    def filter(
        arr1: NumpyFieldList,
        arr2: NumpyFieldList,
        comparison: str,
        threshold: float,
        *,
        replacement: float = 0,
    ) -> NumpyFieldList:
        with ResourceMeter("FILTER"):
            xp = array_api_compat.array_namespace(arr1.values, arr2.values)
            condition = comp_str2func(xp, comparison)(arr2.values, threshold)
            res = xp.where(condition, replacement, arr1.values)
            return FieldList.from_numpy(standardise_output(res), arr1.metadata())

    def pca(
        config,
        ens: NumpyFieldList,
        spread: NumpyFieldList,
        mask: np.ndarray,
        target: str,
    ):
        xp = array_api_compat.array_namespace(ens.values)
        lat_lon = ens.to_latlon(flatten=True)
        ens_data = xp.reshape(
            ens.to_numpy(flatten=True),
            (config.num_members, len(config.steps), len(lat_lon["lat"])),
        )
        with ResourceMeter(
            f"PCA: lat {lat_lon['lat'].shape} lon {lat_lon['lon'].shape}"
        ):
            pca_data = clustereps.pca.do_pca(
                config,
                lat_lon["lat"],
                normalise_angles(lat_lon["lon"]),
                ens_data,
                spread[0].values,
                mask,
            )

        ## Save data
        if target is not None:
            np.savez_compressed(target, **pca_data)

        return (pca_data, ens[0].metadata())

    def cluster(
        config,
        pca_output: tuple[dict, GribBufferMetaData],
        ncomp_file: str,
        indexes: str,
        deterministic: str,
    ):
        pca_data, pca_metadata = pca_output

        ## Compute number of PCs based on the variance threshold
        var_cum = pca_data["var_cum"]
        npc = config.npc
        if npc <= 0:
            npc = clustereps.cluster.select_npc(config.var_th, var_cum)
            if ncomp_file is not None:
                with open(ncomp_file, "w") as f:
                    print(npc, file=f)

        print(f"Number of PCs used: {npc}, explained variance: {var_cum[npc-1]} %")

        with ResourceMeter("Clustering"):
            (
                ind_cl,
                centroids,
                rep_members,
                centroids_gp,
                rep_members_gp,
                ens_mean,
            ) = clustereps.cluster.do_clustering(
                config, pca_data, npc, verbose=True, dump_indexes=indexes
            )

        ## Find the deterministic forecast
        if deterministic is not None:
            with ResourceMeter("Find deterministic"):
                det = read_steps_grib(config.sources, deterministic, config.steps)
                det_index = clustereps.cluster.find_cluster(
                    det,
                    ens_mean,
                    pca_data["eof"][:npc, ...],
                    pca_data["weights"],
                    centroids,
                )
        else:
            det_index = 0

        metadata = pca_metadata.override(
            {"rep_members": rep_members, "det_index": det_index, "ind_cl": ind_cl}
        )
        ret = {
            "centroids": [
                np.array(centroids_gp),
                metadata.override({"type": "cm", "scenario": "centroids"}),
            ],
            "representatives": [
                np.array(rep_members_gp),
                metadata.override({"type": "cr", "scenario": "representatives"}),
            ],
        }
        return ret

    def attribution(config, scenario: str, pca_output: dict, cluster_output: dict):
        pca_data, _ = pca_output
        scdata, metadata = cluster_output[scenario]

        with ResourceMeter("Read climatology"):
            ## Read climatology fields
            clim = clustereps.attribution.get_climatology_fields(
                config.climMeans, config.seasons, config.stepDate
            )

            ## Read climatological EOFs
            clim_eof, clim_ind = clustereps.attribution.get_climatology_eof(
                config.climClusterCentroidsEOF,
                config.climEOFs,
                config.climPCs,
                config.climSdv,
                config.climClusterIndex,
                config.nClusterClim,
                config.monStartDoS,
                config.monEndDoS,
            )

        weights = pca_data["weights"]

        ## Compute anomalies
        anom = scdata - clim
        anom = np.clip(anom, -config.clip, config.clip)

        with ResourceMeter(f"Attribute {scenario}"):
            cluster_att, min_dist = clustereps.attribution.attribution(
                anom, clim_eof, clim_ind, weights
            )

        return (metadata, scdata, anom, cluster_att, min_dist)

    def retrieve(request: dict | list[dict], **kwargs):
        res = ek_retrieve(request, **kwargs)
        ret = FieldList.from_numpy(
            np.asarray(res.values),
            list(map(GribBufferMetaData, res.metadata())),
        )
        return ret

    def write(loc: str, data: NumpyFieldList, grib_sets: dict):
        if loc == "null:":
            return
        target = target_from_location(loc)
        if isinstance(target, (FileTarget, FileSetTarget)):
            # Allows file to be appended on each write call
            target.enable_recovery()
        assert len(data) == 1, f"Expected single field, received {len(data)}"
        metadata = grib_sets.copy()
        metadata.update(data.metadata()[0]._d)
        set_missing = [key for key, value in metadata.items() if value == "MISSING"]
        for missing_key in set_missing:
            metadata.pop(missing_key)

        template = data.metadata()[0].buffer_to_metadata().override(metadata)

        for missing_key in set_missing:
            template._handle.set_missing(missing_key)
        with ResourceMeter(f"WRITE {loc}"):
            write_grib(target, template._handle, data[0].values)

    def cluster_write(
        config,
        scenario,
        attribution_output,
        cluster_dests,
    ):
        metadata, scdata, anom, cluster_att, min_dist = attribution_output
        grib_template = metadata.buffer_to_metadata()
        cluster_type, ind_cl, rep_members, det_index = [
            metadata._d[x] for x in ["type", "ind_cl", "rep_members", "det_index"]
        ]

        keys, steps = get_output_keys(config, grib_template)
        with ResourceMeter(f"WRITE {scenario}"):
            ## Write anomalies and cluster scenarios
            dest, adest = cluster_dests
            target = target_from_location(dest)
            anom_target = target_from_location(adest)
            keys["type"] = cluster_type
            write_cluster_attr_grib(
                steps,
                ind_cl,
                rep_members,
                det_index,
                scdata,
                anom,
                cluster_att,
                target,
                anom_target,
                keys,
                ncl_dummy=config.ncl_dummy,
            )

            ## Write report output
            # table: attribution cluster index for all fc clusters, step
            np.savetxt(
                pjoin(
                    config.output_root,
                    f"{config.step_start}_{config.step_end}dist_index_{scenario}.txt",
                ),
                min_dist,
                fmt="%-10.5f",
                delimiter=3 * " ",
            )

            # table: distance measure for all fc clusters, step
            np.savetxt(
                pjoin(
                    config.output_root,
                    f"{config.step_start}_{config.step_end}att_index_{scenario}.txt",
                ),
                cluster_att,
                fmt="%-3d",
                delimiter=3 * " ",
            )
