from scipy.misc import imresize

import augmentations as aug
import numpy as np
import random
import utils
import os

from data import Data

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

TEST_PATH = 'test'


def test_augmentation():
    r = random.Random(4)

    dataset = Data()
    mpii = dataset.mpii

    batch_num = r.randint(0, mpii.NPY_ESTIMATION_BATCH_COUNT - 1)
    all_labels = mpii.load_npy_labels('est')
    inputs = mpii.load_npy_images(batch_num, 'est')
    labels, all_visible_parts, all_present_parts = mpii.get_labels_in_batch(all_labels, batch_num)

    # DATA AUGMENTATION SETTINGS
    aug_settings = aug.get_random_transform_params(
        input_shape = (mpii.NPY_IMAGES_SIZE, mpii.NPY_IMAGES_SIZE, 3),
        # width_shift_range = 0.01,
        # height_shift_range = 0.05,
        # rotation_range = 30,
        # horizontal_flip = True,
        # zoom_range = (1, 1.25)
    )

    HM_SIZE = 256

    for l, (X, Y, perm) in enumerate(aug.generate_minibatches(
            X = inputs, Y = labels,
            batch_size = 20, rseed = 47 + batch_num, t_params_f = aug_settings,
            final_size = 256, final_heatmap_size = None)):

        for j, img in enumerate(X):
            plt.figure()
            img = np.array(imresize(img, (HM_SIZE, HM_SIZE)) / 256.0)
            plt.imshow(img)

            for i in range(16):
                plt.plot(Y[j, 1, i], Y[j, 0, i], 'g.')

            plt.show()
            plt.savefig(os.path.join(TEST_PATH, 'aug_256_img%02d.png' % j))

        break

test_augmentation()

# def test_heatmaps():
#     r = random.Random(2)
#
#     dataset = Data()
#     mpii = dataset.mpii
#
#     batch_num = r.randint(0, mpii.NPY_ESTIMATION_BATCH_COUNT - 1)
#     all_labels = mpii.load_npy_labels('est')
#     inputs = mpii.load_npy_images(batch_num, 'est')
#     labels, all_visible_parts, all_present_parts = mpii.get_labels_in_batch(all_labels, batch_num)
#
#     X = utils.take_slice(inputs, 0, 5)
#     Y = utils.take_slice(labels, 0, 5)
#
#     desired_heatmaps = nn.generate_output(
#         shape = (count, args.HM_SIZE, args.HM_SIZE, 16),
#         visible_parts = visible_parts,
#         labels = Y, sigma = args.SIGMA)
#
#     for j, img in enumerate(X):
#         plt.figure()
#         img = np.array(imresize(img, (HM_SIZE, HM_SIZE)) / 256.0)
#         plt.imshow(img)
#
#         for i in range(16):
#             plt.plot(Y[j, 1, i], Y[j, 0, i], 'g.')
#
#         plt.show()
#         plt.savefig(os.path.join(TEST_PATH, 'aug_img%02d.png' % j))
#
#     break
#
# test_augmentation()