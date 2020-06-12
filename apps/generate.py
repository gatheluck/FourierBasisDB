import collections
import os
import random
import sys
import tqdm

import hydra
import numpy as np
import omegaconf
import torch
import torchvision

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(base)

from submodules.FourierHeatmap.fhmap import fourier_base


def generate(num_basis: int, num_image_per_class: int, norm: str, num_channel: int, image_size: int):
    """
    generate Fourier Basis DB.
    number of class is decided by the norm of index. currently of l1 and l2 norm are supported.

    Args
    - num_basis: number of Fourier basis used to generate DB. in total, np.floor(num_basis) x 2 types of basis are used.
    - num_image_per_class: number of image to generate per class. Note: currently this variable is not strictly applied.
    - norm: type of norm to use decide class.
    - image_size: size of output image.
    """
    assert num_basis > 0
    assert num_image_per_class > 0
    assert norm in "l1 l2".split()
    assert num_channel > 0
    assert image_size > 0
    assert image_size >= num_basis

    num_basis = num_basis if num_basis % 2 != 0 else num_basis - 1  # num_basis should be odd
    num_basis_half = int(np.floor(num_basis / 2))

    # get indices of Fourier basis function
    indices = [(h_index, w_index) for h_index in range(-num_basis_half, num_basis_half + 1)
                                  for w_index in range(-num_basis_half, num_basis_half + 1)]

    if norm == "l1":
        p = 1
    elif norm == "l2":
        p = 2
    else:
        raise NotImplementedError

    indices_norm = [int(torch.norm(torch.FloatTensor(index), p).item()) for index in indices]
    norm_counts = collections.Counter(indices_norm)

    # create output dir
    root_path = '_'.join(['fbdb', norm, 'basis-{0:04d}'.format(num_basis), 'cls-{0:04d}'.format(len(norm_counts))])
    os.makedirs(root_path)

    # loop over Fourier basis index
    with tqdm.tqdm(indices) as pbar:
        for (h_index, w_index) in pbar:
            # show progress bar
            pbar.set_postfix(collections.OrderedDict(h_index=h_index, w_index=w_index))

            norm = int(torch.norm(torch.FloatTensor([h_index, w_index]), p).item())

            # outpath
            outpath = os.path.join(root_path, '{0:05d}'.format(norm))
            os.makedirs(outpath, exist_ok=True)

            # number of image which is needed
            num_needed_image = int(num_image_per_class * 1.0 / norm_counts[norm])
            if num_needed_image < 1:
                num_needed_image = 1

            # loop over image
            for i_image in range(num_needed_image + 1):

                base_channels = []
                for i in range(num_channel):
                    base = fourier_base.generate_fourier_base(image_size, image_size, h_index, w_index)
                    base = base / base.max().abs()  # scale [-1, 1]

                    eps = random.uniform(-1, 1)
                    base = eps * base  # random rescale

                    base_channels.append(base)

                base = torch.stack(base_channels)  # shape becomes [c, h, w]

                fp = os.path.join(outpath, '{h_index:+04d}_{w_index:+04d}_{i_image:05d}.jpeg'.format(h_index=h_index, w_index=w_index, i_image=i_image))
                torchvision.utils.save_image(base, fp)


@hydra.main(config_path="../conf/generate.yaml")
def main(cfg: omegaconf.DictConfig) -> None:
    print(cfg)
    generate(**cfg)


if __name__ == "__main__":
    main()
