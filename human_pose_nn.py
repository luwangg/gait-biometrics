import io
from abc import abstractmethod

import tensorflow as tf
import numpy as np
import math

from PIL import Image

import settings
import utils
import os

from scipy.stats import multivariate_normal

from model import human_pose_resnet, init_model_variables
from inception_resnet_v2 import inception_resnet_v2_arg_scope, inception_resnet_v2_aux

import tensorflow.contrib.layers as layers
import tensorflow.contrib.distributions as dst
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

slim = tf.contrib.slim

SUMMARY_PATH = '/home/margeta/logdir/'
CHECKPOINT_PATH = os.path.join(settings.DATA_PATH, 'pose_estimation_checkpoints')

KEY_SUMMARIES = tf.GraphKeys.SUMMARIES
KEY_IMAGE_SUMMARIES = 'image_summary'
KEY_HIGH_COST_SUMMARIES = 'high_cost_summary'
KEY_SUMMARIES_PER_JOINT = ['summary_joint_%02d' % i for i in range(16)]
KEY_HIGH_COST_SUMMARIES_PER_JOINT = ['high_cost_summary_joint_%02d' % i for i in range(16)]

class HumanPoseNN(object):

    def __init__(self, name, input_tensor, network, heatmap_size, optimizer_f = None,
                 loss_type = 'MSE', is_train = True):
        self.heatmap_size = heatmap_size
        self.network = network
        self.is_train = is_train
        self.input_tensor = input_tensor
        self.loss_type = loss_type
        self.name = name

        self.sigm_network = tf.sigmoid(self.network)

        self.present_joints = tf.placeholder(
            dtype = tf.float32,
            shape = (None, 16),
            name = 'present_joints')

        self.desired_heatmap = tf.placeholder(
            dtype = tf.float32,
            shape = (None, heatmap_size, heatmap_size, 16),
            name = 'desired_heatmap')

        self.desired_points = tf.placeholder(
            dtype = tf.float32,
            shape = (None, 2, 16),
            name = 'desired_points')

        self.loss_err = self._get_loss_function(loss_type)
        self.euclidean_dist = self.euclidean_dist_err()
        self.euclidean_dist_per_joint = self.euclidean_dist_per_joint_err()

        if is_train:
            self.global_step = tf.Variable(0, name = 'global_step', trainable = False)

            self.learning_rate = tf.placeholder(
                dtype = tf.float32,
                shape = [],
                name = 'learning_rate')

            if optimizer_f is None:
                optimizer_f = tf.train.RMSPropOptimizer
                # optimizer_f = tf.train.AdamOptimizer

            self.optimize = layers.optimize_loss(loss = self.loss_err,
                                                 global_step = self.global_step,
                                                 learning_rate = self.learning_rate,
                                                 optimizer = optimizer_f(self.learning_rate),
                                                 clip_gradients = 2
                                                 )

            # self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, epsilon = 1.0, momentum = 0.9).minimize(self.loss_err)


            # coreVariables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'InceptionResnetV2')
            # var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'InceptionResnetV2')
            # var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'NewInceptionResnetV2')
            #
            # opt1 = tf.train.RMSPropOptimizer(self.learning_rate * 0.2, epsilon = 1.0, momentum = 0.9)
            # opt2 = tf.train.RMSPropOptimizer(self.learning_rate, epsilon = 1.0, momentum = 0.9)
            # # grads = tf.gradients(self.loss_err, var_list1 + var_list2)
            #
            # grads1 = opt1.compute_gradients(self.loss_err, var_list1)
            # grads2 = opt2.compute_gradients(self.loss_err, var_list2)
            # clip_grad1 = [(tf.clip_by_value(grads1, -2., 2.), var) for grad, var in grads1]
            # clip_grad2 = [(tf.clip_by_value(grads2, -2., 2.), var) for grad, var in grads2]
            #
            # # train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
            # # train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
            # train_op1 = opt1.apply_gradients(clip_grad1)
            # train_op2 = opt2.apply_gradients(clip_grad2)
            #
            # train_op = tf.group(train_op1, train_op2)
            #
            # self.optimizer = train_op

            # In order to update moving mean and variance in batch normalization layers
            # self.update_operations = tf.tuple(tf.get_collection(tf.GraphKeys.UPDATE_OPS))


        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())


        # Initialize summaries
        if is_train:
            logdir = os.path.join('/home/margeta/logdir/', self.name, 'train')
            self.summary_writer = tf.train.SummaryWriter(logdir)
            self.summary_writer_by_points = [tf.train.SummaryWriter(os.path.join(logdir, 'point_%02d' % i))
                                             for i in range(16)]

            tf.scalar_summary('Average euclidean distance', self.euclidean_dist, collections = [KEY_SUMMARIES])

            for i in range(16):
                tf.scalar_summary('Joint euclidean distance', self.euclidean_dist_per_joint[i],
                                  collections = [KEY_SUMMARIES_PER_JOINT[i]])

            # for i in range(16):
            #     tf.scalar_summary('Joint heatmap', self.save_heatmap_to_buffer(self.sigm_network[i, ...]),
            #                       collections = [KEY_HIGH_COST_SUMMARIES_PER_JOINT[i]])

            self.create_summary_from_weights()

            self.ALL_SUMMARIES = tf.merge_all_summaries(KEY_SUMMARIES)
            # self.IMAGE_SUMMARIES = tf.merge_all_summaries(KEY_IMAGE_SUMMARIES)
            # self.HIGH_COST_SUMMARIES = tf.merge_all_summaries(KEY_HIGH_COST_SUMMARIES)
            self.SUMMARIES_PER_JOINT = [tf.merge_all_summaries(KEY_SUMMARIES_PER_JOINT[i]) for i in range(16)]
            # self.HC_SUMMARIES_PER_JOINT = [tf.merge_all_summaries(KEY_HIGH_COST_SUMMARIES_PER_JOINT[i]) for i in range(16)]

        print('Model for estimating human pose was loaded!')

    def _get_loss_function(self, loss_type):
        if loss_type == 'MSE':
            loss = self.MSE_loss()
        elif loss_type == 'SCE':
            loss = self.Sigm_CE_loss()
        else:
            raise NotImplementedError('Loss function has to be either MSE or SCE!')

        return loss

    def generate_output(self, shape, presented_parts, labels, sigma):
        res = None

        if self.loss_type == 'MSE':
            res = utils.get_gauss_heat_map(
                shape = shape,
                is_visible = presented_parts,
                mean = labels, sigma = sigma)

        if self.loss_type == 'SCE':
            res = utils.get_binary_heat_map(
                shape = shape,
                is_present = presented_parts,
                centers = labels, diameter = sigma)

        return res

    def MSE_loss(self):

        sq = tf.square(self.sigm_network - self.desired_heatmap)
        loss = tf.reduce_mean(sq)

        # Stop propagation of error of joints that are not present
        # loss = tf.mul(loss, self.present_joints)
        # loss = tf.reduce_sum(loss)

        return loss

    def Sigm_CE_loss(self):
        ce = tf.nn.sigmoid_cross_entropy_with_logits(self.network, self.desired_heatmap)
        err = tf.reduce_mean(ce)

        return err

    def euclidean_dist_err(self):

        # Stop propagation of error of joints that are not present
        l2_dist = tf.mul(self.euclidean_distance(), self.present_joints)
        err = tf.reduce_mean(l2_dist)

        return err

    def euclidean_dist_per_joint_err(self):
        # Stop propagation of error of joints that are not present
        l2_dist = tf.mul(self.euclidean_distance(), self.present_joints)
        err = tf.reduce_mean(l2_dist, reduction_indices = 0)

        return err

    def euclidean_distance(self):
        x = tf.argmax(tf.reduce_max(self.sigm_network, 1), 1)
        y = tf.argmax(tf.reduce_max(self.sigm_network, 2), 1)

        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)

        dy = tf.squeeze(self.desired_points[:, 0, :])
        dx = tf.squeeze(self.desired_points[:, 1, :])

        sx = tf.square(x - dx)
        sy = tf.square(y - dy)

        l2_dist = tf.sqrt(sx + sy)

        return l2_dist

    def feed_forward(self, X):
        out = self.sess.run(self.sigm_network, feed_dict = {
            self.input_tensor: X
        })

        return out

    def test_mse_loss(self, X, Y, present_joints):

        err = self.sess.run(self.loss_err, feed_dict = {
            self.input_tensor: X,
            self.desired_heatmap: Y,
            self.present_joints: present_joints
        })

        return err

    def test_euclidean_distance(self, X, points, present_joints):
        err = self.sess.run(self.euclidean_dist, feed_dict = {
            self.input_tensor: X,
            self.desired_points: points,
            self.present_joints: present_joints
        })

        return err

    @staticmethod
    def get_gauss_pdf_initializer():
        mv = multivariate_normal((0, 0.), (9**2, 9.**2))
        zr = np.linspace(-5, 5, 10)
        X, Y = np.meshgrid(zr, zr)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        gauss_kernel = mv.pdf(pos)
        weights = np.zeros(shape = (10, 10, 16, 16), dtype = np.float32)

        for i in range(16):
            weights[:, :, i, i] = gauss_kernel

        return tf.constant_initializer(value = weights,
                                       dtype = tf.float32)

    @abstractmethod
    def get_network(self, input_tensor, is_training):
        pass

    @abstractmethod
    def create_summary_from_weights(self):
        pass

    @staticmethod
    def save_img_to_buffer(img, point):
        plt.figure()
        plt.imshow(img)
        for i in range(16):
            if 0 < point[1,i] < img.shape[1] and 0 < point[0,i] < img.shape[0]:
                plt.plot(point[1,i], point[0,i], 'g.')

        plt.show()
        buf = io.BytesIO()
        plt.savefig(buf, format = 'png', bbox_inches = 'tight', pad_inches = 0)
        buf.seek(0)
        res_image = np.asarray(Image.open(buf))
        buf.close()
        plt.close()

        return res_image

    @staticmethod
    def save_heatmap_to_buffer(heatmap, point):
        plt.figure()
        plt.imshow(heatmap)
        if 0 < point[1] < heatmap.shape[1] and 0 < point[0] < heatmap.shape[0]:
            plt.plot(point[1], point[0], 'g.')

        plt.show()
        buf = io.BytesIO()
        plt.savefig(buf, format = 'png', bbox_inches = 'tight', pad_inches = 0)
        buf.seek(0)
        res_image = np.asarray(Image.open(buf))
        buf.close()
        plt.close()

        return res_image

    def train(self, X, heatmaps, present_joints, learning_rate):
        if not self.is_train:
            raise Exception('Network is not in train mode!')

        self.sess.run(self.optimize, feed_dict = {
            self.input_tensor: X,
            self.desired_heatmap: heatmaps,
            self.present_joints: present_joints,
            self.learning_rate: learning_rate,
        })

    def write_summary(self, X, Y, heatmaps, present_joints, learning_rate):
        step = tf.train.global_step(self.sess, self.global_step)

        if step % 20 == 0:
            feed_dict = {
                self.input_tensor: X,
                self.desired_points: Y,
                self.desired_heatmap: heatmaps,
                self.present_joints: present_joints,
                self.learning_rate: learning_rate,
            }

            summary, loss = self.sess.run([self.ALL_SUMMARIES, self.loss_err], feed_dict = feed_dict)
            self.summary_writer.add_summary(summary, step)

            if loss < 7e-3:
                loss_summ = tf.Summary()
                loss_summ.value.add(
                    tag = 'Small cross entropy loss',
                    simple_value = float(loss))
                self.summary_writer.add_summary(loss_summ, step)

            # if step % 100 == 0:
            #     high_cost_summary = self.sess.run(self.HIGH_COST_SUMMARIES, feed_dict = feed_dict)
            #     self.summary_writer.add_summary(high_cost_summary)
            #
            #     summaries = self.sess.run(self.HC_SUMMARIES_PER_JOINT, feed_dict = feed_dict)
            #
            #     for i in range(16):
            #         self.summary_writer_by_points[i].add_summary(summaries[i])

            if step % 100 == 0:
                summaries = self.sess.run(self.SUMMARIES_PER_JOINT, feed_dict = feed_dict)

                for i in range(16):
                    self.summary_writer_by_points[i].add_summary(summaries[i], step)

                for i in range(16):
                    self.summary_writer_by_points[i].flush()


                if step % 500 == 0:
                    try:
                        n_images = 8

                        heatmaps = self.sess.run(self.sigm_network, feed_dict = feed_dict)
                        n_images = min(n_images, heatmaps.shape[0])

                        img = tf.image_summary("Pose", np.array([self.save_img_to_buffer(X[i,...], point = Y[i,...]) for i in range(n_images)]),
                                               max_images = n_images)
                        self.summary_writer.add_summary(self.sess.run(img), step)

                        mpl_hm = [tf.image_summary("Pose",
                                                    np.array([self.save_heatmap_to_buffer(heatmaps[j, ..., i], point = Y[j,:,i])
                                   for j in range(n_images)]), max_images = n_images)
                                  for i in range(16)]

                        summaries = self.sess.run(mpl_hm)

                        for i in range(16):
                            self.summary_writer_by_points[i].add_summary(summaries[i], step)
                        for i in range(16):
                            self.summary_writer_by_points[i].flush()
                    except:
                        pass


            self.summary_writer.flush()


class InceptionResnet(HumanPoseNN):
    IMAGE_SIZE = 299
    HEATMAP_SIZE = 289
    SIGMA = 15

    def __init__(self, name, is_training, loss_type = 'MSE'):
        tf.set_random_seed(0)

        input_tensor = tf.placeholder(
            dtype = tf.float32,
            shape = (None, 299, 299, 3),
            name = 'input_image')

        net = self.pre_process(input_tensor)
        net = self.get_network(net, is_training)

        super().__init__(name, input_tensor, net,
                         heatmap_size = 289, loss_type = loss_type, is_train = is_training)

    def restore(self, epoch = None, whole_model = True):
        if whole_model:
            if epoch is None:
                raise Exception('Network epoch number is not given!')

            checkpoint_path = os.path.join(CHECKPOINT_PATH, self.name, 'epoch%02d.ckpt' % epoch)
        else:
            checkpoint_path = os.path.join(CHECKPOINT_PATH, '..', 'inception_resnet_v2_2016_08_30.ckpt')


        all_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope = 'InceptionResnetV2')

        if whole_model:
            all_vars += tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope = 'NewInceptionResnetV2/AuxiliaryScoring')

        saver = tf.train.Saver(all_vars)
        saver.restore(self.sess, checkpoint_path)

    def save(self, epoch):
        checkpoint_path = os.path.join(CHECKPOINT_PATH, self.name)
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        checkpoint_name_path = os.path.join(checkpoint_path, 'epoch%02d.ckpt' % epoch)

        all_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope = 'InceptionResnetV2')
        all_vars += tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope = 'NewInceptionResnetV2/AuxiliaryScoring')

        saver = tf.train.Saver(all_vars)
        saver.save(self.sess, checkpoint_name_path)

    @staticmethod
    def pre_process(inp):
        return ((inp / 255) - 0.5) * 2.0

    @staticmethod
    def get_tconv_filter(f_shape):
        height = f_shape[0]
        width = f_shape[1]

        f = math.ceil(width / 2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)

        bilinear = np.zeros((height, width))
        for x in range(width):
            for y in range(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)

        for i in range(min(f_shape[2], f_shape[3])):
            weights[:, :, i, i] = bilinear

        return tf.constant_initializer(value = weights,
                                       dtype = tf.float32)



    # def get_gauss_pdf(self):
    #
    #     mv = dst.MultivariateNormalDiag(mu = (0, 0.), diag_stdev = (8, 8.))
    #     zr = tf.linspace(-20., 20., 40)
    #     X, Y = tf.meshgrid(zr, zr)
    #     X = tf.expand_dims(X, 2)
    #     Y = tf.expand_dims(Y, 2)
    #
    #     Z = tf.concat(2, [X, Y])
    #
    #     gauss_kernel = tf.reshape(mv.pdf(Z), (40, 40, 1, 1))
    #     gauss_pdf = tf.Variable(gauss_kernel, trainable = False, dtype = tf.float32)
    #
    #     return gauss_pdf

    @abstractmethod
    def get_network(self, input_tensor, is_training):
        pass

    def create_summary_from_weights(self):
        with tf.variable_scope('NewInceptionResnetV2/AuxiliaryScoring', reuse = True):
            tf.histogram_summary('Scoring_layer/biases', tf.get_variable('Scoring_layer/biases'), [KEY_SUMMARIES])
            tf.histogram_summary('Upsampling_layer/biases', tf.get_variable('Upsampling_layer/biases'), [KEY_SUMMARIES])
            tf.histogram_summary('Scoring_layer/weights', tf.get_variable('Scoring_layer/weights'), [KEY_SUMMARIES])
            tf.histogram_summary('Upsampling_layer/weights', tf.get_variable('Upsampling_layer/weights'), [KEY_SUMMARIES])

        with tf.variable_scope('InceptionResnetV2/AuxLogits', reuse = True):
            tf.histogram_summary('Last_layer/weights', tf.get_variable('Conv2d_2a_5x5/weights'), [KEY_SUMMARIES])
            tf.histogram_summary('Last_layer/beta', tf.get_variable('Conv2d_2a_5x5/BatchNorm/beta'), [KEY_SUMMARIES])
            tf.histogram_summary('Last_layer/moving_mean', tf.get_variable('Conv2d_2a_5x5/BatchNorm/moving_mean'), [KEY_SUMMARIES])


class AuxSegmScoringIRNetwork(InceptionResnet):
    def __init__(self, name, is_training, loss_type = 'SCE'):
        super().__init__(name, is_training, loss_type = loss_type)

    def get_network(self, input_tensor, is_training):

        # Load pre-trained inception-resnet model
        with slim.arg_scope(inception_resnet_v2_arg_scope(batch_norm_decay = 0.999, weight_decay = 0.0001)):
            net, end_points = inception_resnet_v2_aux(input_tensor, is_training = is_training)

        net = end_points['BeforeAux']

        # Adding some modification to original InceptionResnetV2 - changing scoring of AUXILIARY TOWER
        with tf.variable_scope('NewInceptionResnetV2'):
            with tf.variable_scope('AuxiliaryScoring'):
                tf.histogram_summary('Last_layer/activations', net, [KEY_SUMMARIES])

                net = slim.dropout(net, 0.8, is_training = is_training, scope = 'Dropout')
                weight_decay = 0.001
                # # Scoring layer
                # net = layers.convolution2d(net, num_outputs = 16, kernel_size = 1, stride = 1,
                #                            scope = 'Scoring_layer', activation_fn = None)

                net = layers.convolution2d(net, num_outputs = 16, kernel_size = 1, stride = 1,
                                           scope = 'Scoring_layer', activation_fn = None,
                                           weights_regularizer = slim.l2_regularizer(weight_decay),
                                           biases_regularizer = slim.l2_regularizer(weight_decay),
                                           weights_initializer = tf.zeros_initializer
                                           # normalizer_fn = slim.batch_norm,
                                           # normalizer_params = {
                                           #      'decay': 0.99,
                                           #      'epsilon': 0.001
                                           #   }
                                           )

                tf.histogram_summary('Scoring_layer/activations', net, [KEY_SUMMARIES])

                # net = layers.convolution2d(net, num_outputs = 450, kernel_size = 1, stride = 1,
                #                            scope = 'Scoring_layer_2', activation_fn = None,
                #                            weights_regularizer = slim.l2_regularizer(weight_decay),
                #                            biases_regularizer = slim.l2_regularizer(weight_decay)
                #                            )

                # Upsampling
                net = layers.convolution2d_transpose(net, num_outputs = 16, kernel_size = 17, stride = 17,
                                                     activation_fn = None, padding = 'VALID',
                                                     scope = 'Upsampling_layer',
                                                     weights_regularizer = slim.l2_regularizer(weight_decay),
                                                     biases_regularizer = slim.l2_regularizer(weight_decay),
                                                     weights_initializer = self.get_tconv_filter((17, 17, 16, 16))
                                                     )

                tf.histogram_summary('Upsampling_layer/activations', net, [KEY_SUMMARIES])

                # net = layers.convolution2d_transpose(net, num_outputs = 16, kernel_size = 17, stride = 17,
                #                                      activation_fn = None, padding = 'VALID', trainable = False,
                #                                      weights_initializer = self.get_tconv_filter((17, 17, 16, 16)))


            # Smoothing layer
            # net = layers.convolution2d(net, num_outputs = 16, kernel_size = 10,
            #                            stride = 1, padding = 'SAME',
            #                            activation_fn = None, scope = 'Smoothing_layer',
            #                            trainable = False, biases_initializer = None,
            #                            weights_initializer = self.get_gauss_pdf_initializer())

            return net

class AuxScoringIRNetwork(InceptionResnet):
    def __init__(self, name, is_training, loss_type = 'SCE'):
        super().__init__(name, is_training, loss_type = loss_type)

    def get_network(self, input_tensor, is_training):

        # Load pre-trained inception-resnet model
        with slim.arg_scope(inception_resnet_v2_arg_scope(batch_norm_decay = 0.999, weight_decay = 0.0001)):
            net, end_points = inception_resnet_v2_aux(input_tensor, is_training = is_training)

        # Adding some modification to original InceptionResnetV2 - changing scoring of AUXILIARY TOWER
        with tf.variable_scope('NewInceptionResnetV2'):
            with tf.variable_scope('AuxiliaryScoring'):
                tf.histogram_summary('Last_layer/activations', net, [KEY_SUMMARIES])

                net = slim.dropout(net, 0.6, is_training = is_training, scope = 'Dropout')
                weight_decay = 0.0005
                # # Scoring layer
                # net = layers.convolution2d(net, num_outputs = 16, kernel_size = 1, stride = 1,
                #                            scope = 'Scoring_layer', activation_fn = None)

                net = layers.convolution2d(net, num_outputs = 512, kernel_size = 1, stride = 1,
                                           scope = 'Scoring_layer', activation_fn = None,
                                           weights_regularizer = slim.l2_regularizer(weight_decay),
                                           biases_regularizer = slim.l2_regularizer(weight_decay),
                                           # weights_initializer = tf.zeros_initializer
                                           # normalizer_fn = slim.batch_norm,
                                           # normalizer_params = {
                                           #      'decay': 0.99,
                                           #      'epsilon': 0.001
                                           #   }
                                           )

                tf.histogram_summary('Scoring_layer/activations', net, [KEY_SUMMARIES])

                # net = layers.convolution2d(net, num_outputs = 450, kernel_size = 1, stride = 1,
                #                            scope = 'Scoring_layer_2', activation_fn = None,
                #                            weights_regularizer = slim.l2_regularizer(weight_decay),
                #                            biases_regularizer = slim.l2_regularizer(weight_decay)
                #                            )

                # Upsampling
                net = layers.convolution2d_transpose(net, num_outputs = 16, kernel_size = 17, stride = 17,
                                                     activation_fn = None, padding = 'VALID',
                                                     scope = 'Upsampling_layer',
                                                     weights_regularizer = slim.l2_regularizer(weight_decay),
                                                     biases_regularizer = slim.l2_regularizer(weight_decay),
                                                     )

                tf.histogram_summary('Upsampling_layer/activations', net, [KEY_SUMMARIES])

                # net = layers.convolution2d_transpose(net, num_outputs = 16, kernel_size = 17, stride = 17,
                #                                      activation_fn = None, padding = 'VALID', trainable = False,
                #                                      weights_initializer = self.get_tconv_filter((17, 17, 16, 16)))


            # Smoothing layer
            net = layers.convolution2d(net, num_outputs = 16, kernel_size = 10,
                                       stride = 1, padding = 'SAME',
                                       activation_fn = None, scope = 'Smoothing_layer',
                                       trainable = False, biases_initializer = None,
                                       weights_initializer = self.get_gauss_pdf_initializer())

            return net


class ResidualNet(HumanPoseNN):
    IMAGE_SIZE = 256
    HEATMAP_SIZE = 256
    SIGMA = 11

    def __init__(self, name, is_training, loss_type = 'SCE'):
        tf.set_random_seed(0)

        input_tensor = tf.placeholder(
            dtype = tf.float32,
            shape = (None, 256, 256, 3),
            name = 'input_image')

        net = self.pre_process(input_tensor)
        net = self.get_network(net, is_training)

        super().__init__(name, input_tensor, net,
                         heatmap_size = 256, loss_type = loss_type, is_train = is_training)

    @staticmethod
    def pre_process(inp):
        return inp / 255

    def restore(self, epoch = None, whole_model = True):
        if whole_model:
            if epoch is None:
                raise Exception('Network epoch number is not given!')

            checkpoint_path = os.path.join(CHECKPOINT_PATH, self.name, 'epoch%02d.ckpt' % epoch)

            all_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope = 'HumanPoseResnet')
            all_vars += tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope = 'NewHumanPoseResnet/Scoring')

            saver = tf.train.Saver(all_vars)
            saver.restore(self.sess, checkpoint_path)
        else:
            init_model_variables(os.path.join(settings.DATA_PATH, 'hp.t7'), trainable = True)

    def save(self, epoch):
        checkpoint_path = os.path.join(CHECKPOINT_PATH, self.name)
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        checkpoint_name_path = os.path.join(checkpoint_path, 'epoch%02d.ckpt' % epoch)

        all_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope = 'HumanPoseResnet')
        all_vars += tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope = 'NewHumanPoseResnet/Scoring')

        saver = tf.train.Saver(all_vars)
        saver.save(self.sess, checkpoint_name_path)

    def create_summary_from_weights(self):
        pass
        # with tf.variable_scope('NewHumanPoseResnet/Scoring', reuse = True):
        #     tf.histogram_summary('Scoring_layer/biases', tf.get_variable('Scoring_layer/biases'), [KEY_SUMMARIES])
        #     tf.histogram_summary('Upsampling_layer/biases', tf.get_variable('Upsampling_layer/biases'), [KEY_SUMMARIES])
        #     tf.histogram_summary('Scoring_layer/weights', tf.get_variable('Scoring_layer/weights'), [KEY_SUMMARIES])
        #     tf.histogram_summary('Upsampling_layer/weights', tf.get_variable('Upsampling_layer/weights'), [KEY_SUMMARIES])
        #
        # with tf.variable_scope('HumanPoseResnet/Block_4/Bottleneck_2', reuse = True):
        #     tf.histogram_summary('Last_layer/weights', tf.get_variable('Conv_3/weights'), [KEY_SUMMARIES])
        #     tf.histogram_summary('Last_layer/beta', tf.get_variable('BatchNorm_3/beta'), [KEY_SUMMARIES])
        #     tf.histogram_summary('Last_layer/gamma', tf.get_variable('BatchNorm_3/gamma'), [KEY_SUMMARIES])
        #     tf.histogram_summary('Last_layer/moving_mean', tf.get_variable('BatchNorm_3/moving_mean'), [KEY_SUMMARIES])
        #     tf.histogram_summary('Last_layer/moving_variance', tf.get_variable('BatchNorm_3/moving_variance'), [KEY_SUMMARIES])

    def get_network(self, input_tensor, is_training):
        net_end, end_points = human_pose_resnet(input_tensor, reuse = True, training = is_training)
        return net_end
        net = end_points['resnet_end']

        with tf.variable_scope('NewHumanPoseResnet'):
            with tf.variable_scope('Scoring'):
                tf.histogram_summary('Last_layer/activations', net, [KEY_SUMMARIES])

                # net = slim.dropout(net, 1.0, is_training = is_training, scope = 'Dropout')
                weight_decay = 0.001

                # Scoring layer
                net = layers.convolution2d(net, num_outputs = 512, kernel_size = 1, stride = 1,
                                           scope = 'Scoring_layer', activation_fn = None,
                                           weights_regularizer = slim.l2_regularizer(weight_decay),
                                           biases_regularizer = slim.l2_regularizer(weight_decay)
                                           )

                tf.histogram_summary('Scoring_layer/activations', net, [KEY_SUMMARIES])

                # Upsampling
                net = layers.convolution2d_transpose(net, num_outputs = 16, kernel_size = 16, stride = 16,
                                                     activation_fn = None, padding = 'VALID',
                                                     scope = 'Upsampling_layer',
                                                     weights_regularizer = slim.l2_regularizer(weight_decay),
                                                     biases_regularizer = slim.l2_regularizer(weight_decay)
                                                     )

                tf.histogram_summary('Upsampling_layer/activations', net, [KEY_SUMMARIES])

            # Smoothing layer
            net = layers.convolution2d(net, num_outputs = 16, kernel_size = 10,
                                       stride = 1, padding = 'SAME',
                                       activation_fn = None, scope = 'Smoothing_layer',
                                       trainable = False, biases_initializer = None,
                                       weights_initializer = self.get_gauss_pdf_initializer())

            return net