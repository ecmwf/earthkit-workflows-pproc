import copy

# Request formatting symbols
LIST_SEPARATOR = "/"
KV_SEPARATOR = "="
NEW_LINE = ","
RANGE_INDICATOR = "to"


class RequestType:
    COMPUTE = "compute"


class ProductType:
    PERTURBED_FORECAST = "pf"
    CTRL_FORECAST = "cf"
    DET_FORECAST = "fc"
    ENS_MEAN = "em"
    ENS_STD = "es"
    EFI = "efi"
    EFI_CONTROL = "efic"
    SOT = "sot"
    QUANTILES = "pb"
    EVENT_PROB = "ep"
    CLUSTER_MEAN = "cm"
    CLUSTER_REP = "cr"
    CLUSTER_STD = "cs"


class MarsKey:
    PARAM = "param"
    TYPE = "type"
    STEP = "step"
    GRID = "grid"
    TARGET = "target"
    EXPVER = "expver"
    STREAM = "stream"
    DATE = "date"
    TIME = "time"
    NUMBER = "number"
    LEVTYPE = "levtype"
    LEVELIST = "levelist"
    CLASS = "class"
    DOMAIN = "domain"


class ComputeRequest:
    def __init__(
        self,
        param: str,
        param_type: str,
        steps: list,
        grid: str,
        target: str,
        base_request: dict,
    ):
        self.param = param
        self.type = param_type
        if isinstance(steps[0], int):
            self.steps = [steps]
        else:
            self.steps = [x.split("-") if "-" in x else [x, x] for x in steps]
        self.grid = grid
        self.target = target
        self.base_request = base_request


def expand(value_list):
    if RANGE_INDICATOR in value_list:
        interval_range = value_list.split(LIST_SEPARATOR)
        if len(interval_range) == 3:
            start, _, end = interval_range
            return [int(start), int(end)]
        start, _, end, _, interval = interval_range
        return [int(start), int(end), int(interval)]
    return value_list.split(LIST_SEPARATOR)


def format_request(request):
    ret = copy.deepcopy(request)
    for k, v in ret.items():
        if k == MarsKey.TARGET:
            continue
        if LIST_SEPARATOR in v or k in [MarsKey.PARAM, MarsKey.TYPE]:
            ret[k] = expand(v)
    return ret


def new_requests(request_type: str, request: dict):
    mod_request = format_request(request)
    params = mod_request.pop(MarsKey.PARAM)
    grid = mod_request.pop(MarsKey.GRID, None)
    steps = mod_request.pop(MarsKey.STEP)
    target = mod_request.pop(MarsKey.TARGET)
    if request_type == RequestType.COMPUTE:
        types = mod_request.pop(MarsKey.TYPE)
        for param in params:
            for type in types:
                yield ComputeRequest(param, type, steps, grid, target, mod_request)
    else:
        raise ValueError(
            f"Unknown request type {request_type}. Expected one of {RequestType}"
        )


def parse_request(filename: str):
    requests = []
    with open(filename, "r") as file:
        new_request = None
        for line in file:
            if new_request is None:
                request_type = line.split(NEW_LINE)[0]
                new_request = {}
            else:
                kv_pairs = line.split(NEW_LINE)
                for pair in kv_pairs:
                    try:
                        key, value = pair.split(KV_SEPARATOR)
                        new_request[key.lstrip(" ").lstrip("\t")] = value.rstrip("\n")
                    except ValueError:
                        pass
                if NEW_LINE not in line:
                    # Create request and append to requests
                    requests.extend(new_requests(request_type, new_request))
                    new_request = None
    return requests
