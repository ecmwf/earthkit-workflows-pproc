import pytest

from ppcascade.entry.genconfig import RequestTranslator


@pytest.fixture
def generate_request(param: int, param_type: str, extra_keys: str):
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
	step=0/to/24/by/6,               
	target=fileset:test.grib,{extra_keys}
	type={param_type}
"""


@pytest.mark.parametrize(
    "param, param_type, extra_keys, prod",
    [
        [130, "em/es", "", "ensms"],
        [10, "fc/cf", "", "forecast"],
        [10, "pf", "number=1/to/5", "forecast"],
        [131060, "ep", "", "prob"],
        [132228, "efi/efic", "", "extreme"],
        [132228, "sot", "number=10/90,", "extreme"],
        [130, "pb", "quantile=100,", "quantiles"],
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
