"""Exercise real generation and preview commands without external data."""
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]


def cli(script, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / 'apps' / script), *map(str, args)],
        cwd=ROOT / 'apps', env={**os.environ, 'HYDRA_FULL_ERROR': '1'},
        capture_output=True, text=True, timeout=120,
    )


@pytest.mark.parametrize('preview_count', [1, 7])
@pytest.mark.parametrize('metric', ['freq', 'index', 'balance'])
def test_generate_and_preview(tmp_path, metric, preview_count):
    run = tmp_path / 'generate'
    result = cli('generate.py', f'metric={metric}', 'num_basis=3', 'image_size=8',
                 'num_image_per_class=2', f'hydra.run.dir={run}')
    assert result.returncode == 0, result.stderr
    datasets = list(run.glob('fbdb_*'))
    assert len(datasets) == 1
    pictures = list(datasets[0].glob('*/*.jpeg'))
    assert pictures
    for picture in pictures:
        with Image.open(picture) as image:
            assert image.size == (8, 8)
            assert image.mode == 'RGB'
    preview = tmp_path / 'preview'
    result = cli('show_sample.py', f'db_path={datasets[0]}', f'num_image_per_class={preview_count}',
                 f'hydra.run.dir={preview}')
    assert result.returncode == 0, result.stderr
    with Image.open(preview / 'sample_image.png') as image:
        assert image.width == 8 * len(list(datasets[0].iterdir()))
        assert image.height == 8 * preview_count


def test_reject_invalid_basis_without_dataset(tmp_path):
    result = cli('generate.py', 'num_basis=2', f'hydra.run.dir={tmp_path}')
    assert result.returncode != 0
    assert '[num_basis] option should be odd' in result.stderr
    assert not list(tmp_path.glob('fbdb_*'))


def test_reject_missing_database(tmp_path):
    result = cli('show_sample.py', f'hydra.run.dir={tmp_path}')
    assert result.returncode != 0
    assert 'please specify db_path' in result.stderr
    assert not (tmp_path / 'sample_image.png').exists()


def test_validation_split_preserves_all_generated_images(tmp_path):
    result = cli('generate.py', 'metric=index', 'num_basis=3', 'image_size=8',
                 'num_image_per_class=3', 'val_ratio=0.25', f'hydra.run.dir={tmp_path}')
    assert result.returncode == 0, result.stderr
    dataset, = tmp_path.glob('fbdb_*')
    train = list((dataset / 'train').glob('*/*.jpeg'))
    val = list((dataset / 'val').glob('*/*.jpeg'))
    # Six basis classes, each historically generates requested count + one.
    assert len(train) == 18
    assert len(val) == 6
    assert set(p.name for p in train).isdisjoint(p.name for p in val)
    assert {p.name for p in dataset.iterdir()} == {'train', 'val'}
