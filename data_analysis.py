# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import tensorflow as tf
import os
import io
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt

def main(_):
    train_folder = 'train'
    folders = os.walk(train_folder)
    all_sizes = []
    for folder_content in folders:
        print(folder_content[0])
        for filename in folder_content[2]:
            if filename.endswith('.png'):
                full_path = os.path.join(folder_content[0], filename)
                # print(full_path)
                image = PIL.Image.open(full_path)
                all_sizes.append(image.size)

    x = [e[0] for e in all_sizes]
    y = [e[1] for e in all_sizes]
    # print(x)
    print(np.average(x), np.average(y))

    # plt.scatter(x, y, s=5)
    # plt.show()


if __name__ == '__main__':
  tf.app.run()
