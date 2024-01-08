import pytest

from ppcascade.entry.genconfig import RequestTranslator


@pytest.fixture
def generate_request(param: int, param_type: str, steps: str, extra_keys: str):
    return f"""compute,
    class=od,
    domain=g,
    expver=0001,
    levelist=250/500/850,
    levtype=pl,
    stream=enfo,
    date=20230914,            
    time=12,                 
    param={param},               
	grid=O640,                
	step={steps},               
	target=fileset:test.grib,{extra_keys}
	type={param_type}
"""


@pytest.mark.parametrize(
    "param, param_type, steps, extra_keys, prod",
    [
        [130, "em/es", "0/to/24/by/6", "", "EnsmsConfig"],
        [10, "fc/cf", "0/to/24/by/6", "", "ForecastConfig"],
        [10, "pf", "0/to/24/by/6", "number=1/to/5", "ForecastConfig"],
        [131060, "ep", "0/to/24/by/6", "", "ProductConfig"],
        [131022, "ep", "0/6/12", "", "EnsembleAnomalyConfig"],
        [131022, "ep", "0/to/24/by/6", "", "EnsembleAnomalyConfig"],
        [132228, "efi/efic", "12-24/24-36", "", "ExtremeConfig"],
        [132228, "sot", "12-24/24-36", "number=10/90,", "ExtremeConfig"],
        [132144, "efi", "12-24/24-36", "", "ExtremeConfig"],
        [130, "pb", "0/to/24/by/6", "quantile=100,", "QuantileConfig"],
    ],
)
def test_config_generation(generate_request, prod, request, tmpdir):
    conf = RequestTranslator(f"{request.config.rootpath}/configs/config.yaml")
    request_file = f"{tmpdir}/request"
    with open(request_file, "w") as f:
        f.write(generate_request)

    products = conf.translate(request_file)
    assert len(products) == 1
    assert list(products.keys())[0] == prod
