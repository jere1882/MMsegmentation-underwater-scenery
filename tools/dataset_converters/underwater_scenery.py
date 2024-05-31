import argparse
import os
import shutil

import cv2
import numpy as np

from scipy.io import loadmat

def parse_args():
    parser = argparse.ArgumentParser(
        description=\
        'Convert underwater scenery annotations to mmsegmentation format')  # noqa
    parser.add_argument('underwater_scenery_datast_path', help='underwater scenery data path')
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--nproc', default=16, type=int, help='number of process')
    args = parser.parse_args()
    return args

color2index = {
    (255, 255, 255) : 255,  #255
    (0,     0,   0) : 0, # 0
    (0,     0, 255) : 1,  #29
    (0,   255, 255) : 2,  #178 o 179
    (0,   255,   0) : 3,  #149 o 150
    (255, 255,   0) : 4,  #225 o 226
    (255,   0, 255) : 5,  #105
    (255,   0,   0) : 6,  # 76
}

def main():
    args = parse_args()
    dataset_path = args.underwater_scenery_datast_path
    nproc = args.nproc

    out_dir = args.out_dir or dataset_path

    train_annotations = [ out_dir+'train/masks/'+fn for fn in os.listdir(os.path.join(out_dir, 'train/masks')) ]
    val_annotations = [ out_dir+'val/masks/'+fn for fn in os.listdir(os.path.join(out_dir, 'val/masks')) ]

    all_annotations = train_annotations + val_annotations

    for idx, img in enumerate(all_annotations):
        if img.endswith(".bmp"):
            print(idx)
            new_path = img.replace('.bmp', '.png')
            image = cv2.imread(img, cv2.IMREAD_COLOR)

            lut = np.ones(256, dtype=np.uint8) * 255
            lut[[255,29,179,150,226,105,76]] = np.arange(7, dtype=np.uint8)
            im_out = cv2.LUT(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), lut)
            cv2.imwrite(new_path, im_out)


if __name__ == '__main__':
    main()