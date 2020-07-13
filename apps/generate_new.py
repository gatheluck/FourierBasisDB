import collections
import os
import random
import sys

import hydra
import numpy as np
import omegaconf
import torch
import torchvision
import tqdm

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(base)

from fbdb.utils import val_prep
from submodules.FourierHeatmap.fhmap import fourier_base


def generate(
    num_basis: int,
    num_image_per_class: int,
    metric: str,
    norm_type: str,
    num_channel: int,
    image_size: int,
    val_ratio: float = 0.0,
    log_dir: str = ".",
):
    """
    generate Fourier Basis DB.
    number of class is decided by the norm of index. currently of l1 and l2 norm are supported.

    Args
    - num_basis: number of Fourier basis used to generate DB. in total, np.floor(num_basis) x 2 types of basis are used.
    - num_image_per_class: number of image to generate per class. Note: currently this variable is not strictly applied.
    - metric: (IMPORTANT) this option decides the metric which is used to create classes.
    - norm_type: (IMPORTANT) norm type. norm info might be used when grouping and freqency blancing process.
    - image_size: size of output image.
    - val_ratio: ratio of validatation set. if not zero, apply val_prep function.
    - log_dir: log directory for dataset.
    """
    SUPPORTED_METRIC = set("freq index balance".split())
    SUPPORTED_NORM = set("l2 l1".split())

    assert num_basis > 0, "[num_basis] option should be larger than 0."
    assert num_basis % 2 != 0, "[num_basis] option should be odd."
    assert (
        num_image_per_class > 0
    ), "[num_image_per_class] option should be larger than 0."
    assert metric in SUPPORTED_METRIC, "[grouping_metric] option is invalid."
    assert norm_type in SUPPORTED_NORM, "[norm_type] option is invalid."
    assert image_size > 0, "[image_size] option should be larger than 0."
    assert image_size >= num_basis
    assert 0 <= val_ratio < 1.0

    num_basis_half = int(np.floor(num_basis / 2))  # this is even value.

    # get indices of Fourier basis function
    indices = [
        (h_index, w_index)
        for h_index in range(-num_basis_half, num_basis_half + 1)
        for w_index in range(
            0, num_basis_half + 1
        )  # IMPORTANT: both (i,j) and (-i,-j) create same Fourier Heatmap. So we only loop over [0, num_basis_half] about width index.
    ]

    if norm_type == "l1":
        p = 1
    elif norm_type == "l2":
        p = 2
    else:
        raise NotImplementedError

    # list of frequency info of each index (i,j).
    indices_freq = [
        int(torch.norm(torch.FloatTensor(index), p).item()) for index in indices
    ]
    # gouping by freq and counts elements of each group.
    freq_counts = collections.Counter(indices_freq)

    num_classes = len(freq_counts) if metric == "freq" else len(indices)

    # create output dir
    dataset_name = "_".join(
        [
            "fbdb",
            "metric-{metric}".format(metric=metric),
            "norm-{norm}".format(norm=norm_type),
            "basis-{0:04d}".format(num_basis),
            "cls-{0:04d}".format(num_classes),
        ]
    )
    root_path = os.path.join(log_dir, dataset_name)
    os.makedirs(root_path)

    # loop over Fourier basis index
    with tqdm.tqdm(indices) as pbar:
        for (h_index, w_index) in pbar:
            # show progress bar
            pbar.set_postfix(collections.OrderedDict(h_index=h_index, w_index=w_index))

            freq = int(torch.norm(torch.FloatTensor([h_index, w_index]), p).item())

            # outpath
            outpath = (
                os.path.join(root_path, "{0:05d}".format(freq))
                if metric == "freq"
                else os.path.join(
                    root_path, "{0:+04d}_{1:+04d}".format(h_index, w_index)
                )
            )
            os.makedirs(outpath, exist_ok=True)

            # number of image which is needed
            num_needed_image = (
                num_image_per_class
                if metric == "index"
                else int(
                    num_image_per_class * 1.0 / freq_counts[freq]
                )  # for 'balance' or 'freq'
            )
            if num_needed_image < 1:
                num_needed_image = 1

            # loop over image
            for i_image in range(num_needed_image + 1):

                base_channels = []
                for i in range(num_channel):
                    base = fourier_base.generate_fourier_base(
                        image_size, image_size, h_index, w_index
                    )
                    base = base / base.max().abs()  # scale [-1, 1]

                    eps = random.uniform(-1, 1)
                    base = eps * base  # random rescale to [-eps, eps]

                    base_channels.append(base)

                base = torch.stack(base_channels)  # shape becomes [c, h, w]

                fp = os.path.join(
                    outpath,
                    "{h_index:+04d}_{w_index:+04d}_{i_image:05d}.jpeg".format(
                        h_index=h_index, w_index=w_index, i_image=i_image
                    ),
                )
                torchvision.utils.save_image(base, fp)

    # apply val prep
    if val_ratio > 0.0:
        val_prep(dataset_path=root_path, val_ratio=val_ratio, keep_orig=False)


@hydra.main(config_path="../conf/generate.yaml")
def main(cfg: omegaconf.DictConfig) -> None:
    print(cfg)
    generate(**cfg)


if __name__ == "__main__":
    main()
