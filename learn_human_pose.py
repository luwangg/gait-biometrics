import itertools
import random

from scipy.misc import imsave
from scipy.stats import multivariate_normal

import augmentations as aug
import numpy as np
import data
import settings
import utils
import math
import os

from data import Data
from human_pose_nn import InceptionResnet, AuxScoringIRNetwork, ResidualNet, AuxSegmScoringIRNetwork
from model import init_model_variables
from stopwatch import toc, tic

# OPTIONS
EPOCHS = 10
MINIBATCH_SIZE = 8


# INITIALIZE DATASET -> MPII + LSP
dataset = Data()
mpii = dataset.mpii
mpii.set_350_cached_data()

# DATA AUGMENTATION SETTINGS
aug_settings = aug.get_random_transform_params(
    input_shape = (mpii.NPY_IMAGES_SIZE, mpii.NPY_IMAGES_SIZE, 3),
    width_shift_range = 0.01,
    height_shift_range = 0.05,
    rotation_range = 30,
    horizontal_flip = True,
    zoom_range = (0.90, 1.25)
)

# INITIALIZE NN
# tic()
# print('Loading weights ...')
# init_model_variables(os.path.join(settings.DATA_PATH, 'hp.t7'), trainable = True)
# print('Weights was loaded')
# toc()

tic()
print('Loading model ...')
# nn = ResidualNet('02_resnet', is_training = True)
nn = AuxScoringIRNetwork('01_IR', is_training = True)
print('Model was loaded')
toc()

tic()
print('Loading weights ...')
# path = os.path.join(settings.DATA_PATH, 'pose_estimation_checkpoints', args.NET_NAME, 'epoch%02d.ckpt' % 7)
# nn.restore(path)

nn.restore(whole_model = False)
# nn.restore('./inception_resnet_v2_2016_08_30.ckpt', whole_model = False)

print('Weights was loaded')
toc()

# RUN TRAINING PROCESS
r = random.Random(1)
batch_perm = list(range(mpii.NPY_ESTIMATION_BATCH_COUNT))
all_est_labels = mpii.load_npy_labels('est')
all_val_labels = mpii.load_npy_labels('val')

alpha = 1e-3
alpha_d = 0.93

for epoch in range(1, EPOCHS+1):
    print('Epoch %d' % epoch)

    alpha_l = alpha*alpha_d**(epoch-1)
    print('learning rate: ', alpha_l)
    r.shuffle(batch_perm)

    # TRAINING NETWORK
    print('Start training')
    for batch_num in batch_perm:
        inputs = mpii.load_npy_images(batch_num, 'est')
        labels, b_visible_parts, b_present_parts = mpii.get_labels_in_batch(all_est_labels, batch_num)

        print('batch num:', batch_num)

        tic()
        for l, (X, Y, perm) in enumerate(aug.generate_minibatches(
                X = inputs, Y = labels,
                batch_size = MINIBATCH_SIZE, rseed = epoch * 47 + batch_num, t_params_f = aug_settings,
                final_size = nn.IMAGE_SIZE, final_heatmap_size = nn.HEATMAP_SIZE)):

            present_parts = np.array([b_present_parts[p] for p in perm], dtype = np.float32)
            visible_parts = np.array([b_visible_parts[p] for p in perm])
            count = X.shape[0]

            desired_heatmaps = nn.generate_output(
                shape = (count, nn.HEATMAP_SIZE, nn.HEATMAP_SIZE, 16),
                presented_parts = present_parts,
                labels = Y, sigma = nn.SIGMA)

            nn.train(X, desired_heatmaps, present_parts, alpha_l)
            nn.write_summary(X, Y, desired_heatmaps, present_parts, alpha_l)
        toc()

    # SAVE MODEL
    nn.save(epoch)
    print("Model was saved!")