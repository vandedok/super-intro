import os
import shutil as sh
from pathlib import Path
from itertools import product
import numpy as np
import cv2


def get_ranges(dim_size, window_size, stride):
    starts = np.arange(0, dim_size, stride)
    stops = starts + window_size
    if stops[-1] > dim_size:
        stops[-1] = dim_size
    ranges = np.stack([starts, stops]).T
    return ranges

def makedir_overwrite(dir, overwrite):
    if os.path.exists(dir):
        if overwrite:
            sh.rmtree(dir)
        else:
            raise Exception("Directory exists and overwirte mode is disabled, aborting.")
    os.makedirs(dir, exist_ok=True)

def compose_window_name(x0, y0, x1, y1, image_xy_str, window_size_str, ch):
    name = "_".join(
        [str(i) for i in (x0, y0, x1, y1)] +\
        image_xy_str +\
        [ch]+\
        window_size_str,
        )
    name += ".png"

    return name

def parse_window_name(window_name):
    name = window_name.split(".")[0]
    return [int(x) for x in name.split("_")]


def split(image, output_dir, window_size, strides, overwrite):
    
    output_dir = Path(output_dir)
    for s, w, d in zip(strides, window_size, ("x","y")):
        if s > w:
            raise ValueError("Stride is greater than window for dim %s. Reconstruction may not be possible" %d)
    
    makedir_overwrite(output_dir, overwrite)

    image_xy = (image.shape[1], image.shape[0])
    image_xy_str = [str(d) for d in image_xy]
    window_size_str = [str(d) for d in window_size]
    ch_str = str(image.shape[-1])
    ranges = [get_ranges(l, w, s) for l, w, s in zip(image_xy, window_size, strides)]

    for (x0, x1), (y0, y1) in product(*ranges):
        window = image[y0:y1, x0:x1]
        name = compose_window_name(x0, y0, x1, y1, image_xy_str, window_size_str, ch_str)
        cv2.imwrite(
            str(output_dir / name),
            window
        )

    


def merge(input_dir):
    input_dir_path = Path(input_dir)
    windows_paths = list(input_dir_path.glob("*"))
    n = len(windows_paths)
    _, _, _, _, image_x, image_y,ch, window_x, window_y = parse_window_name(windows_paths[0].name)
    image = np.zeros((int(image_y), int(image_x), ch), dtype=np.uint8)

    for path in windows_paths:
        x0, y0, x1, y1, im_x, im_y,ch, w_x, w_y = parse_window_name(path.name)
        if im_x != image_x or im_y != image_y or w_x != window_x or w_y != window_y:
            raise ValueError("Problems with image parames in the following filename: \n %s"%path.name)
        image[y0:y1, x0:x1] = cv2.imread(str(path))

    return image


def check_split_merge(    
    image_path,
    output_dir,
    window_size,
    strides,
    ):

    image = cv2.imread(image_path)
    split(
        image = image,

        output_dir = output_dir,
        window_size = window_size,
        strides = strides,
        overwrite =False
    )

    new_image = merge(output_dir)
    sh.rmtree(output_dir)

    return np.prod(new_image==image)