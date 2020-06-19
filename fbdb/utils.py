import os
import sys
import glob
import random
import shutil
import tqdm
import collections


base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)


def val_prep(dataset_path: str, val_ratio: float = 0.1, keep_orig: bool = False):
    """
    sparate val set from training set. new train and val set is placed under dataset_path/train and dataseth_path/val.
    if keep_orig flag is False, original dataset is removed.

    Args:
    - dataset_path: path to original dataset.
    - val_ratio: ratio of validation dataset.
    - keep_orig: flag wheather keeps original dataset or not.
    """
    assert 0.0 < val_ratio < 1.0

    classpaths = glob.glob(os.path.join(dataset_path, '*'))
    if not classpaths:
        raise ValueError('dataset_path is empty')

    # create output dirs
    trainpath = os.path.join(dataset_path, 'train')
    valpath = os.path.join(dataset_path, 'val')
    os.makedirs(trainpath)
    os.makedirs(valpath)

    with tqdm.tqdm(enumerate(classpaths)) as pbar:
        for class_idx, classpath in pbar:
            pbar.set_postfix(collections.OrderedDict(class_idx=class_idx))

            # check classpath
            imgpaths = glob.glob(os.path.join(classpath, '*'))
            random.shuffle(imgpaths)

            if not imgpaths:
                raise ValueError('class is empty')

            # create class output dir under train and val
            classname = os.path.basename(classpath)
            os.makedirs(os.path.join(trainpath, classname))
            os.makedirs(os.path.join(valpath, classname))

            # loop over images
            for img_idx, imgpath in enumerate(imgpaths):
                imgname = os.path.basename(imgpath)
                if img_idx < int(len(imgpaths) * val_ratio):  # val
                    shutil.copy(imgpath, os.path.join(valpath, classname, imgname))
                else:  # train
                    shutil.copy(imgpath, os.path.join(trainpath, classname, imgname))

            # remove images in original path
            if not keep_orig:
                shutil.rmtree(classpath)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    val_prep(dataset_path=args.path)
