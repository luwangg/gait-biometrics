import numpy as np
import tensorflow as tf

import matplotlib as mpl

import settings
from args_human_pose import NET_NAME
from human_pose_nn import AuxScoringIRNetwork
import os

from models.slim.nets.inception_resnet_v2 import inception_resnet_v2_arg_scope, inception_resnet_v2

mpl.use('Agg')
import matplotlib.pyplot as plt

slim = tf.contrib.slim

print('Load model graph')
nn = AuxScoringIRNetwork(is_training = False)
print('Load model weights')
# nn.restore(os.path.join(settings.DATA_PATH, 'inception_resnet_v2_2016_08_30.ckpt'), whole_model = False)
path = os.path.join(settings.DATA_PATH, 'pose_estimation_checkpoints', NET_NAME, 'epoch%02d.ckpt' % 1)
nn.restore(path)

# with tf.variable_scope('InceptionResnetV2', reuse = True):
#     vm = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope = 'InceptionResnetV2')
lstm_variables = [tf.Print(v, v.name)
                  for v in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope = 'InceptionResnetV2')
                  if 'bias' in v.name]

print(nn.sess.run(lstm_variables))

# plt.figure()
# plt.imshow(bm)
# plt.show()
# plt.savefig('000.png')


