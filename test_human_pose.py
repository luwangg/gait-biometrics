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
from human_pose_nn import InceptionResnet, AuxScoringIRNetwork#, ResidualNet
from stopwatch import toc, tic


TEST_BATCH_COUNT = 100
EPOCHS = list(range(1,11))

# INITIALIZE NN
tic()
print('Loading Inception resnet model ...')
nn = AuxScoringIRNetwork(is_training = False)
print('Inception resnet model was loaded')
toc()

for EPOCH in EPOCHS:
    print('CHECKPOINT %02d' % EPOCH)

    tic()
    print('Loading Inception resnet weights ...')
    path = os.path.join(settings.DATA_PATH, 'pose_estimation_checkpoints', args.NET_NAME, 'epoch%02d.ckpt' % EPOCH)
    nn.restore(path)
    print('Inception resnet weights was loaded')
    toc()

    # INITIALIZE DATASET -> MPII + LSP
    dataset = Data()
    mpii = dataset.mpii

    batch_perm = list(range(args.ESTIMATION_BATCH_COUNT))
    val_perm = list(range(args.VALIDATION_BATCH_COUNT))
    all_est_labels = mpii.load_npy_labels('est')
    all_val_labels = mpii.load_npy_labels('val')

    net_path = os.path.join(settings.DATA_PATH, 'pose_estimation_checkpoints', args.NET_NAME)
    f_est_l2_err = open(os.path.join(net_path, 'est_l2_e%02d.txt' % EPOCH), 'w')
    f_val_l2_err = open(os.path.join(net_path, 'val_l2_e%02d.txt' % EPOCH), 'w')
    f_est_mse_err = open(os.path.join(net_path, 'est_mse_e%02d.txt' % EPOCH), 'w')
    f_val_mse_err = open(os.path.join(net_path, 'val_mse_e%02d.txt' % EPOCH), 'w')

    # VALIDATE NETWORK
    val_l2_err = 0
    est_l2_err = 0
    val_mse_err = 0
    est_mse_err = 0

    r = utils.to_int((args.IMG_AUG_WIDTH - args.IMG_SIZE) / 2)
    mr = utils.to_int((args.IMG_AUG_WIDTH - args.HM_SIZE) / 2)

    # ESTIMATION ERRs
    # print('Start testing estimation err')
    # for batch_num in batch_perm:
    #     print('batch num:', batch_num)
    #     inputs = mpii.load_npy_images(batch_num, 'est')
    #     labels, all_visible_parts, all_present_parts = mpii.get_labels_part(all_est_labels, batch_num)
    #
    #     inputs = inputs[:, r:r + args.IMG_SIZE, r:r + args.IMG_SIZE, :]
    #     labels -= mr
    #
    #     tic()
    #     for i in range(utils.to_int(math.ceil(inputs.shape[0] / TEST_BATCH_COUNT))):
    #         X = utils.take_slice(inputs, i, TEST_BATCH_COUNT)
    #         Y = utils.take_slice(labels, i, TEST_BATCH_COUNT)
    #         visible_parts = utils.take_slice(all_visible_parts, i, TEST_BATCH_COUNT)
    #         present_parts = utils.take_slice(all_present_parts, i, TEST_BATCH_COUNT)
    #
    #         desired_heatmaps = nn.generate_output(
    #             shape = (X.shape[0], args.HM_SIZE, args.HM_SIZE, 16),
    #             is_visible = visible_parts,
    #             mean = labels, sigma = args.SIGMA)
    #
    #         est_mse_err += nn.test_mse_loss(X, desired_heatmaps, present_parts)
    #         est_l2_err += nn.test(X, Y, present_parts)
    #     toc()
    #
    # print(est_mse_err, file = f_est_mse_err)
    # print(est_l2_err, file = f_est_l2_err)

    # VALIDATION ERRs
    print('Start testing validation err')
    val_data_count = 0

    for batch_num in val_perm:
        print('batch num:', batch_num)
        inputs = mpii.load_npy_images(batch_num, 'val')
        labels, all_visible_parts, all_present_parts = mpii.get_labels_in_batch(all_val_labels, batch_num)

        inputs = inputs[:, r:r + args.IMG_SIZE, r:r + args.IMG_SIZE, :]
        labels -= mr

        input_count = inputs.shape[0]
        val_data_count += input_count

        tic()
        for i in range(utils.to_int(math.ceil(input_count / TEST_BATCH_COUNT))):
            X = utils.take_slice(inputs, i, TEST_BATCH_COUNT)
            Y = utils.take_slice(labels, i, TEST_BATCH_COUNT)
            visible_parts = utils.take_slice(all_visible_parts, i, TEST_BATCH_COUNT)
            present_parts = utils.take_slice(all_present_parts, i, TEST_BATCH_COUNT)

            count = X.shape[0]

            desired_heatmaps = nn.generate_output(
                shape = (count, args.HM_SIZE, args.HM_SIZE, 16),
                visible_parts = visible_parts,
                labels = labels, sigma = args.SIGMA)

            val_mse_err += nn.test_mse_loss(X, desired_heatmaps, present_parts)
            val_l2_err += nn.test_euclidean_distance(X, Y, present_parts)
        toc()

    val_mse_err /= val_data_count * 16
    val_l2_err /= val_data_count * 16

    print(val_mse_err, file = f_val_mse_err)
    print(val_l2_err, file = f_val_l2_err)