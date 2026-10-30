"""Keep the documented output location when upgrading Hydra on Python 3.12."""
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name', ['generate', 'show_sample'])
def test_output_is_created_in_hydra_run_directory(name):
    with initialize_config_dir(version_base='1.3', config_dir=str(ROOT / 'conf')):
        cfg = compose(config_name=name, return_hydra_config=True)
    assert cfg.hydra.job.chdir is True


def test_generate_overrides_still_reach_application():
    with initialize_config_dir(version_base='1.3', config_dir=str(ROOT / 'conf')):
        cfg = compose(config_name='generate', overrides=['num_basis=3', 'image_size=8', 'metric=index'])
    assert (cfg.num_basis, cfg.image_size, cfg.metric) == (3, 8, 'index')
