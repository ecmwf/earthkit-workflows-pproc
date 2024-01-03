from ppcascade.entry.parsemars import parse_request
from ppcascade.entry.genconfig import RequestTranslator


def test_config_generation(tmpdir):
    requests = parse_request(
        "/etc/ecmwf/nfs/dh1_home_a/mawj/Documents/earthkit-graph/request"
    )
    print(requests)
    conf = RequestTranslator(
        "/home/mawj/Documents/earthkit-graph/configs/config.yaml"
    )
    for product, prod_conf in conf.translate(requests).items():
        prod_conf.to_yaml(f"{tmpdir}/test_{product}.yaml")
