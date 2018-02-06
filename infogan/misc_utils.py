import random

from os.path import exists, join
from os import listdir, walk

from PIL import Image

import numpy as np
import progressbar



OPS =  [
   ('+', lambda a, b: a+b),
   ('-', lambda a, b: a-b),
   ('*', lambda a, b: a*b),
   ('x', lambda a, b: a*b),
   ('/', lambda a, b: a//b),
]


def parse_math(s):
   for operator, f in OPS:
       try:
           idx = s.index(operator)
           return f(parse_math(s[:idx]), parse_math(s[idx+1:]))
       except ValueError:
           pass
   return int(s)

def next_unused_name(name):
    save_name = name
    name_iteration = 0
    while exists(save_name):
        save_name = name + "-" + str(name_iteration)
        name_iteration += 1
    return save_name


def add_boolean_cli_arg(parser, name, default=False, help=None):
    parser.add_argument(
        "--%s" % (name,),
        action="store_true",
        default=default,
        help=help
    )
    parser.add_argument(
        "--no%s" % (name,),
        action="store_false",
        dest=name
    )


def create_progress_bar(message):
    widgets = [
        message,
        progressbar.Counter(),
        ' ',
        progressbar.Percentage(),
        ' ',
        progressbar.Bar(),
        progressbar.AdaptiveETA()
    ]
    pbar = progressbar.ProgressBar(widgets=widgets)
    return pbar

# yield 是一个类似 return 的关键字，迭代一次遇到yield时就返回yield后面(右边)的值。重点是：下一次迭代时，从上一次迭代遇到的yield后面的代码(下一行)开始执行。
# 简要理解：yield就是 return 返回一个值，并且记住这个返回的位置，下次迭代就从这个位置后(下一行)开始。
# 该函数实质功能就是将path路径下，后缀名为extensions的文件返回（完整的路径）
def find_files_with_extension(path, extensions):
    for basepath, directories, fnames in walk(path):
        for fname in fnames:
            name = fname.lower()
            if any(name.endswith(ext) for ext in extensions):
                yield join(basepath, fname)


# 该函数的主要作用为将特定路劲下的文件（数据集名）中的.png,.jpg,.jpeg
# 随机取Max_images张，归一化，resize到特定大小，存到storage中，返回。
def load_image_dataset(path,
                       desired_height=None,
                       desired_width=None,
                       value_range=None,
                       max_images=None,
                       force_grayscale=False):
    image_paths = list(find_files_with_extension(path, [".png", ".jpg", ".jpeg"]))
    limit_msg = ''
    if max_images is not None and len(image_paths) > max_images:
        # 在得到的图像集中随机选择需要数量的图像（前提是图像集中图像的数量大于需要的图像数据）
        image_paths = random.sample(image_paths, max_images)
        limit_msg = " (limited to %d images by command line argument)" % (max_images,)

    print("Found %d images in %s%s." % (len(image_paths), path, limit_msg))
    #显示进度条而已
    pb = create_progress_bar("Loading dataset ")


    storage = None

    image_idx = 0
    for fname in pb(image_paths):
        # image = Image.open(join(path, fname))
        # image_paths中包含了图像的路径，不用join
        image = Image.open(fname)
        width, height = image.size
        if desired_height is not None and desired_width is not None:
            if width != desired_width or height != desired_height:
                image = image.resize((desired_width, desired_height), Image.BILINEAR)
        else:
            desired_height = height
            desired_width = width

        if force_grayscale:
            image = image.convert("L")

        img = np.array(image)
        # print(type(img))  #显示类型
        # print(img.shape)  #显示尺寸
        # print(img.shape[0])  #图片宽度
        # print(img.shape[1])  #图片高度
        # print(img.shape[2])  #图片通道数
        if len(img.shape) == 2:
            # extra channel for grayscale images
            img = img[:, :, None]

        if storage is None:
            storage = np.empty((len(image_paths), img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)

        storage[image_idx] = img

        image_idx += 1

    if value_range is not None:
        storage = (
            value_range[0] + (storage / 255.0) * (value_range[1] - value_range[0])
        )
    print("dataset loaded.", flush=True)
    return storage

