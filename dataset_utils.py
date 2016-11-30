import data as d
import numpy as np
import utils
import torchfile as tr
import pickle
import os
from stopwatch import tic, toc

data = d.Data()
mpii = data.mpii

def save_labels():
    val_mpii_arr = list(mpii.get_cropped_labels(mpii.NPY_IMAGES_SIZE, mpii.val_perm))
    est_mpii_arr = list(mpii.get_cropped_labels(mpii.NPY_IMAGES_SIZE, mpii.est_perm))

    est_labels = np.array([q[0] for q in est_mpii_arr])
    est_visible = np.array([q[1] for q in est_mpii_arr])
    est_present = np.array([q[2] for q in est_mpii_arr])

    val_labels = np.array([q[0] for q in val_mpii_arr])
    val_visible = np.array([q[1] for q in val_mpii_arr])
    val_present = np.array([q[2] for q in val_mpii_arr])

    np.save('/home/data/DATA/MPII/processed/pose/val_labels', val_labels)
    np.save('/home/data/DATA/MPII/processed/pose/val_visible_joints', val_visible)
    np.save('/home/data/DATA/MPII/processed/pose/val_present_joints', val_present)

    np.save('/home/data/DATA/MPII/processed/pose/est_labels', est_labels)
    np.save('/home/data/DATA/MPII/processed/pose/est_visible_joints', est_visible)
    np.save('/home/data/DATA/MPII/processed/pose/est_present_joints', est_present)


def load_labels():
    labels = np.load('outputs/mpii_labels.npy')
    is_visible = np.load('outputs/mpii_visible_joints.npy')
    is_present = np.load('outputs/mpii_present_joints.npy')


def save_images():
    for i, imgs_arr in enumerate(
            utils.take_slices(data.mpii.get_cropped_frames(mpii.NPY_IMAGES_SIZE, data.mpii.est_perm), mpii.NPY_BATCH_SIZE)):
        imgs = np.array(imgs_arr)
        print('est batch:', i)
        np.save('/home/data/DATA/MPII/processed/images/est_part%02d' % i, imgs)

    for i, imgs_arr in enumerate(
            utils.take_slices(data.mpii.get_cropped_frames(mpii.NPY_IMAGES_SIZE, data.mpii.val_perm), mpii.NPY_BATCH_SIZE)):
        imgs = np.array(imgs_arr)
        print('val batch:', i)
        np.save('/home/data/DATA/MPII/processed/images/val_part%02d' % i, imgs)


def load_images():
    return np.load('/home/data/DATA/MPII/processed/mpii_images_part00.npy')


def load_mpii_torch():
    f = tr.load('/home/solution/lua/mpii_dataset.t7')

    mpii = d.Data().mpii
    mpii.load()

    # map = {info.image.name: i for i, info in enumerate(mpii.labels.annolist)}

    val = [x for x in f if x[b'type'] == 0]
    est = [x for x in f if x[b'type'] == 2]

    val_perm = { x[b'image'].decode("utf-8") for x in val }
    est_perm = { x[b'image'].decode("utf-8") for x in est }

    with open('/home/data/DATA/MPII/val_perm.pickle', 'wb') as handle:
        pickle.dump(val_perm, handle)

    with open('/home/data/DATA/MPII/est_perm.pickle', 'wb') as handle:
        pickle.dump(est_perm, handle)

def create_resnet_output_features():
    pass



# with open('/home/data/DATA/MPII/est_perm.pickle', 'rb') as handle:
#   a = pickle.load(handle)
#
# with open('/home/data/DATA/MPII/val_perm.pickle', 'rb') as handle:
#   b = pickle.load(handle)

print('saving labels')
save_labels()
#
# print('saving images')
# save_images()