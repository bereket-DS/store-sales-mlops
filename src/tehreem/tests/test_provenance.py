import yaml

from src.tehreem.utils import provenance, read_params


def test_monitoring_settings_do_not_invalidate_model_provenance(tmp_path):
    params = read_params()
    initial = provenance()['source_sha256']
    params['monitoring']['reference_days'] = 28
    path = tmp_path / 'params.yaml'
    path.write_text(yaml.safe_dump(params))
    assert provenance(path)['source_sha256'] == initial
    params['training']['lightgbm']['num_leaves'] = 127
    path.write_text(yaml.safe_dump(params))
    assert provenance(path)['source_sha256'] != initial
