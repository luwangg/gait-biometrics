import os

import settings
import utils
from data import Data
# from human_pose_nn import InceptionResnet, AuxScoringIRNetwork

import matplotlib as mpl

from human_pose_nn import AuxScoringIRNetwork, ResidualNet
from model import init_model_variables

mpl.use('Agg')
import matplotlib.pyplot as plt



dataset = Data()
mpii = dataset.mpii

# print('Load model graph')
# nn = AuxScoringIRNetwork('run1', is_training = False)
init_model_variables(os.path.join(settings.DATA_PATH, 'hp.t7'), trainable = False)
nn = ResidualNet('resnet1', is_training = False)
# print('Load model weights')
# path = os.path.join(settings.DATA_PATH, 'pose_estimation_checkpoints', NET_NAME, 'epoch%02d.ckpt' % 2)
# nn.restore(whole_model = False)


all_labels = mpii.load_npy_labels('est')
inputs = mpii.load_npy_images(0, 'est')
labels, all_visible_parts, all_present_parts = mpii.get_labels_in_batch(all_labels, 0)

print(labels.shape)

# r = utils.to_int((350 - 299) / 2)
# mr = utils.to_int((350 - 289) / 2)
#
# inputs = inputs[:, r:r + 299, r:r + 299, :]
# labels -= mr

X = utils.take_slice(inputs, 0, 5)
Y = nn.feed_forward(X)

path = os.path.join(settings.SOLUTION_PATH, 'images')

# Save image
plt.figure()
plt.imshow(X[0,...])
plt.show()
plt.savefig(os.path.join(path, 'im1.jpg'))

# Save heatmap
for i in range(16):
    plt.figure()
    plt.imshow(Y[0,:,:,i])
    plt.plot(labels[0,1,i], labels[0,0,i], 'g.')
    plt.show()
    plt.savefig(os.path.join(path, 'im1h%02d.png' % i))

print('Done')