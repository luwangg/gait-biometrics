import utils
import numpy as np

from scipy import ndimage
from utils import to_int


def get_random_transform_params(input_shape, rotation_range = 0., height_shift_range = 0., width_shift_range = 0.,
                                shear_range = 0., zoom_range = (1, 1), horizontal_flip = False):
    """
    This closure function returns generative function. The returned function gets random instance trough parameter and
    together with closed input parameters generates random parameters for transformation matrix.

    :param input_shape: Shape of images to be transformed with matrix with this parameters
    :param rotation_range: Interval of rotation in degrees (used for in both direction)
    :param height_shift_range: Value of two-sided interval of random shift in vertical direction
    :param width_shift_range: Value of two-sided interval of random shift in horizontal direction
    :param shear_range: Value of two-sided interval of random shear in horizontal direction
    :param zoom_range: Tuple with 2 values representing range of random zoom (values > 1.0 is for zoom out)
    :param horizontal_flip: Whether do random horisontal flip image
    :return: Function that with given random instance generates random parameters for transformation matrix

    """

    def get_instance(rnd):
        U = rnd.uniform

        rr = rotation_range
        hs = height_shift_range
        ws = width_shift_range
        sr = shear_range

        return {
            'input_shape': input_shape,
            'theta': np.pi / 180 * U(-rr, rr) if rr else 0,
            'ty': U(-hs, hs) * input_shape[0] if hs else 0,
            'tx': U(-ws, ws) * input_shape[1] if ws else 0,
            'shear': U(-sr, sr) if shear_range else 0,
            'z': rnd.uniform(zoom_range[0], zoom_range[1]) if zoom_range != (1, 1) else 1,
            'h_flip': rnd.rand() < 0.5 if horizontal_flip else False,
        }

    return get_instance


def assemble_transformation_matrix(input_shape, theta = 0, tx = 0, ty = 0, shear = 0, z = 1):
    """
    Creates transformation matrix with given parameters. That resulting matrix has origin in centre of the image

    :param input_shape: Shape of images to be transformed with matrix. Origin of transformation matrix is set
                        in the middle of image.

    :param theta: Rotation in radians
    :param tx: Translation in X axis
    :param ty: Translation in Y axis
    :param shear: Shear in horizontal direction
    :param z: Image zoom
    :return: Transformation matrix
    """

    def transform_matrix_offset_center(matrix, x, y):
        """
        Creates translation matrix from input matrix with origin in the centre of image

        :param matrix: Input matrix
        :param x: Width of the image
        :param y: Height of the image
        :return: Returns shifted input matrix with origin in [y/2, x/2]
        """
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        t_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return t_matrix

    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    translation_matrix = np.array([[1, 0, ty],
                                   [0, 1, tx],
                                   [0, 0, 1]])

    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    zoom_matrix = np.array([[z, 0, 0],
                            [0, z, 0],
                            [0, 0, 1]])

    # Assembling transformation matrix
    transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

    # Set origin of transformation to center of the image
    h, w = input_shape[0], input_shape[1]
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)

    return transform_matrix


def transform(v, t_matrix, h_flip = False):
    """
    Transform image with (inverted) transformation matrix

    :param v: Input image to be transformed
    :param t_matrix: Transformation matrix
    :param h_flip: Whether do horizontal flip
    :return: Transformed image
    """

    def apply_transform(x, transform_matrix, channel_index = 0, fill_mode = 'nearest', cval = 0.):
        x = np.rollaxis(x, channel_index, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]
        channel_images = [ndimage.interpolation.affine_transform(x_channel, final_affine_matrix,
                                                                 final_offset, order = 2, mode = fill_mode,
                                                                 cval = cval)
                          for x_channel in x]
        x = np.stack(channel_images, axis = 0)
        x = np.rollaxis(x, 0, channel_index + 1)
        return x

    def flip_axis(x, axis):
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x

    v = apply_transform(v, t_matrix, 2)

    if h_flip:
        v = flip_axis(v, 1)

    return v


def crop_data(img, labels, new_img_size, new_label_size = None):
    """
    Both images and labels will be cropped to match the given size

    :param img: Images to be cropped
    :param labels: Labels to be cropped
    :param new_img_size: New image size
    :param new_label_size: New labels size
    :return: Cropped image and labels
    """

    img_size = img.shape[0]

    r = to_int((img_size - new_img_size) / 2)

    img = img[r:r + new_img_size, r:r + new_img_size, :]
    labels -= r

    if new_label_size is not None:
        labels = np.array((labels / new_img_size) * new_label_size, dtype = np.int32)

    return img, labels


def flip_body_joints(points):
    """
    Change semantic of labels after flip transformation - i.e. left leg will be now right and so on.

    :param points: Body joints to be changed
    """

    def swap(a, b):
        points[:,[a,b]] = points[:,[b,a]]

    # Leg
    swap(0, 5)
    swap(1, 4)
    swap(2, 3)

    # Arm
    swap(10, 15)
    swap(11, 14)
    swap(12, 13)


def generate_minibatches(X, Y = None, batch_size = 32, rseed = 0,
                         final_size = None, t_params_f = None, final_heatmap_size = None):
    """
    This function splits whole input batch of images into minibatches of given size. All images in batch are
    transformed using affine transformations in order to prevent over-fitting during training.

    :param X: Batch of input images to be divided. It has to be 4D tensor [batch, channel, height, width]
    :param Y: Labels of input images (joint positions on heatmap). 3D tensor [batch, image dimension, joint].
              E.g. joint with index 4 present in 10th image (i.e. index 9) that is in position [50, 80] is in
              indexes: Y[9, :, 4] == [50, 80]
    :param batch_size: Size of each mini-batch
    :param rseed: Random seed
    :param t_params_f: Function that generates parameters for transformation matrix (see get_random_transform_params)
    :param final_size: Transformed images are cropped to match the given size
    :param final_heatmap_size: Size of heatmaps
    :return: Sequence of randomly ordered and transformed mini-batches
    """

    rnd = np.random.RandomState(rseed)

    if not t_params_f:
        raise Exception('No attributes given!')

    if final_size is None:
        final_size = min(X.shape[2], X.shape[3])

    n = X.shape[0]
    perm = rnd.permutation(n)

    for idx in range(0, n, batch_size):
        b = perm[idx:min(idx + batch_size, n)]

        X_t = []
        Y_t = []

        for k in b:
            t_params = t_params_f(rnd)
            h_flip = t_params.pop('h_flip')
            t_matrix = assemble_transformation_matrix(**t_params)

            x_t = transform(np.squeeze(X[k]), t_matrix, h_flip)
            y_t = utils.get_affine_transform(np.squeeze(Y[k]), np.linalg.inv(t_matrix)) if Y is not None else None

            x_t, y_t = crop_data(x_t, y_t, final_size, final_heatmap_size)

            X_t.append(x_t)
            if Y is not None:
                if h_flip:
                    y_t[1, :] = (final_size if final_heatmap_size is None else final_heatmap_size) - y_t[1, :]
                    flip_body_joints(y_t)

                Y_t.append(y_t)

        if Y is not None:
            yield np.array(X_t), np.array(Y_t), b
        else:
            yield np.array(X_t), b
