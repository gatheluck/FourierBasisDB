import os
import sys
import hydra
import omegaconf
import glob
import random
import torch
import torchvision
import PIL

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)


@hydra.main(config_path='../conf/show_sample.yaml')
def main(cfg: omegaconf.DictConfig) -> None:
    if cfg.db_path == '':
        raise ValueError('please specify db_path option')

    classpaths = sorted(glob.glob(os.path.join(cfg.db_path, '*')))
    sample_images = []

    for classpath in classpaths:
        imgpaths = glob.glob(os.path.join(classpath, '*' + cfg.image_ext))

        if len(imgpaths) == 0:
            raise ValueError('some class does not have any image')

        if len(imgpaths) > cfg.num_image_per_class:
            selected_imgpaths = random.sample(imgpaths, cfg.num_image_per_class)
        else:
            selected_imgpaths = selected_imgpaths + [selected_imgpaths[-1] for i in range(cfg.num_image_per_class - len(imgpaths))]

        class_sample_images = []
        for imgpath in selected_imgpaths:
            img = PIL.Image.open(imgpath)
            img = torchvision.transforms.ToTensor()(img)
            class_sample_images.append(img)

        sample_images.append(torch.cat(class_sample_images, dim=1))

    sample_images = torch.cat(sample_images, dim=2)
    torchvision.utils.save_image(sample_images, 'sample_image.png')


if __name__ == '__main__':
    main()