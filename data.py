import re
import os
import sys
import pickle
from functools import lru_cache
from sys import platform

# import cv2
import abc
import numpy as np
from abc import abstractmethod

from scipy.io import loadmat
from scipy.misc import imresize

import settings
from pictures import load_image
from utils import slice_pad, get_gauss_heat_map, to_int


class Data(object):
    """
        Responsibility of this class is to fetch and manipulate
        the data for:
            1. gait recognition in .jpg sequences or .avi video format
            2. pose estimation from still images
            3. pose estimation from video sequence

        After initialization, property 'gait_datasets' should be initialized by:
            CASIA A,
            CASIA B,
            CASIA C

        :param datasets: List of all datasets, where each dataset is represented as a list of Persons
        :param Data.DATA_PATH: Path to the dataset files

        :type datasets: List[List[Person]]
    """

    # Setting location of the data
    if platform == 'win32':
        DATA_PATH = 'D:\DIPLOMOVKA\DATASETS\DATA'
    else:
        DATA_PATH = settings.DATA_PATH

    # Location of datasets
    _CASIA_PATH = os.path.join(DATA_PATH, 'CASIA')
    _H36M_PATH = os.path.join(DATA_PATH, 'H3.6M')
    _MPII_PATH = os.path.join(DATA_PATH, 'MPII')

    class _DatasetInfo(object):
        @abstractmethod
        def get_persons(self):
            pass

    class _PoseDatasetInfo(object):
        @abstractmethod
        def get_images(self):
            pass

    class _CasiaA(_DatasetInfo):
        def __init__(self):
            self.PATH = os.path.join(Data._CASIA_PATH, 'DatasetA', 'gaitdb')
            self.BACKGROUND_PATH = os.path.join(Data._CASIA_PATH, 'DatasetA', 'bkimages')
            self.WIDTH = 352
            self.HEIGHT = 240

        def get_persons(self):
            for person in os.listdir(self.PATH):
                person_path = os.path.join(self.PATH, person)

                sequences = (os.path.join(person_path, seq)
                             for seq in os.listdir(person_path)
                             if re.search('^00', seq))

                seqs = [ImageGaitSequence(
                            sequence_id = os.path.basename(seq),
                            path = seq,
                            w = self.WIDTH,
                            h = self.HEIGHT,
                            background_path = os.path.join(
                                self.BACKGROUND_PATH,
                                "{0}-{1}-bk.png".format(person, os.path.basename(seq)))
                        ) for seq in sequences]

                yield Person(
                        person_id = person,
                        sequences = seqs)

    class _CasiaB(_DatasetInfo):
        # Name of persons in which sequences is presented noisy shadows
        seq_with_shadows = {'005'}

        def __init__(self):
            self.PATH = os.path.join(Data._CASIA_PATH, 'DatasetB', 'videos')
            self.WIDTH = 320
            self.HEIGHT = 240

        def get_persons(self):
            persons = {}

            for video_file in os.listdir(self.PATH):
                name = video_file[0:3]
                status = video_file[4:6]
                sequence = video_file[7:9]
                angle = video_file[10:13]

                # Reject background sequences and non 90 deg angle
                if int(angle) != 90 or status == 'bk':
                    continue

                if name not in persons:
                    persons[name] = Person(name)

                seq = VideoGaitSequence(
                    sequence_id=sequence,
                    path = os.path.join(self.PATH, video_file),
                    background_path = os.path.join(self.PATH, "{0}-bkgrd-{1}.avi".format(name, angle)),
                    w = self.WIDTH,
                    h = self.HEIGHT,
                    angle=int(angle),
                    has_bag = (status == 'bg'),
                    has_clothes = (status == 'cl'),
                    has_shadow = name in self.seq_with_shadows
                )

                person = persons[name]
                person.add_sequence(seq)

            return persons.values()

    class _CasiaC(_DatasetInfo):
        def __init__(self):
            self.PATH = os.path.join(Data._CASIA_PATH, 'DatasetC', 'videos')
            self.WIDTH = 277
            self.HEIGHT = 190

        def get_persons(self):
            persons = {}

            for video_file in os.listdir(self.PATH):
                name = video_file[2:5]
                status = video_file[5:7]
                sequence = video_file[7:9]

                if status == 'bn':
                    continue

                if name not in persons:
                    persons[name] = Person(name)

                seq = VideoGaitSequence(
                    sequence_id = sequence,
                    path = os.path.join(self.PATH, video_file),
                    background_path = os.path.join(self.PATH, "01{0}bn00.avi".format(name, sequence)),
                    is_infrared = True,
                    w = self.WIDTH,
                    h = self.HEIGHT,
                    has_bag = (status == 'fb'),
                    frame_clip = (20, 210, 18, 295),
                    speed = 'fast'   if (status == 'fq') else
                            'slow'   if (status == 'fs') else
                            'normal'
                )

                person = persons[name]
                person.add_sequence(seq)

            return persons.values()

    class _H36m(_DatasetInfo):

        def __init__(self):
            self.PATH = Data._H36M_PATH
            self.PATH_VIDEOS = 'Videos'
            self.PATH_SILHOUETTES = os.path.join('MySegmentsMat', 'ground_truth_bs')
            self.PATH_LABELS = 'MyPoseFeatures'
            self.PATH_2D_POSITIONS = os.path.join(self.PATH_LABELS, 'D2_Positions')
            self.PATH_3D_POSITIONS = os.path.join(self.PATH_LABELS, 'D3_Positions_mono')

            self.HEIGHT = 1000
            self.WIDTH = 1000

            self.TRAINING_SET = [
                'S1', 'S5', 'S7', 'S8', 'S9'
            ]

            self.VALIDATION_SET = [
                'S6', 'S11'
            ]

        @lru_cache()
        def get_persons(self):
            persons = {}

            for action in os.listdir(self.PATH):
                action_path = os.path.join(self.PATH, action)

                for person_name in os.listdir(action_path):
                    person_path = os.path.join(action_path, person_name)

                    videos_path = os.path.join(person_path, self.PATH_VIDEOS)
                    positions_2D_path = os.path.join(person_path, self.PATH_2D_POSITIONS)
                    positions_3D_path = os.path.join(person_path, self.PATH_3D_POSITIONS)
                    silhouettes_path = os.path.join(person_path, self.PATH_SILHOUETTES)

                    if person_name not in persons:
                        persons[person_name] = Person(person_name)

                    for video in os.listdir(videos_path):
                        if video.endswith('.mp4'):
                            name = video[0:-4]

                            sequence = PoseSequence(
                                sequence_id = name,
                                path = os.path.join(videos_path, video),
                                h = self.HEIGHT,
                                w = self.WIDTH,
                                positions_2D_path = os.path.join(positions_2D_path, name + '.mat'),
                                positions_3D_path = os.path.join(positions_3D_path, name + '.mat'),
                                silhouettes_path = os.path.join(silhouettes_path, name + '.mp4'),
                                action = action
                            )

                            persons[person_name].add_sequence(sequence)

            values = persons.values()
            values = list(values)

            values.sort(key = lambda x: x.person_id)

            return values

    class _MPII(_PoseDatasetInfo):

        def __init__(self):
            self.PATH = os.path.join(Data._MPII_PATH, 'images')
            self.PATH_POSE = os.path.join(Data._MPII_PATH, 'pose', 'mpii_human_pose.mat')
            self.labels = None

            # Properties of cached processed images - centered and scaled around person
            self.NPY_PATH_IMAGES = os.path.join(Data._MPII_PATH, 'processed', 'images')
            self.NPY_PATH_POSE = os.path.join(Data._MPII_PATH, 'processed', 'pose')
            self.NPY_BATCH_SIZE = 2000
            self.NPY_IMAGES_SIZE = 256
            self.NPY_ESTIMATION_BATCH_COUNT = 12
            self.NPY_VALIDATION_BATCH_COUNT = 4

            self._est_perm = None
            self._val_perm = None

        def set_350_cached_data(self):
            self.NPY_PATH_IMAGES = os.path.join(Data._MPII_PATH, 'processed_350', 'images')
            self.NPY_PATH_POSE = os.path.join(Data._MPII_PATH, 'processed_350', 'pose')

            self.NPY_IMAGES_SIZE = 350


        @property
        def est_perm(self):
            """
            Set of indexes of data that belongs to estimation data set
            """
            if self._est_perm is None:
                with open(os.path.join(Data._MPII_PATH, 'est_perm.pickle'), 'rb') as handle:
                    self._est_perm = pickle.load(handle)

            return self._est_perm

        @property
        def val_perm(self):
            """
            Set of indexes of data that belongs to validation data set
            """
            if self._val_perm is None:
                with open(os.path.join(Data._MPII_PATH, 'val_perm.pickle'), 'rb') as handle:
                    self._val_perm = pickle.load(handle)

            return self._val_perm

        def load(self):
            """
            Load matlab file with labels provided with dataset
            """
            if self.labels is None:
                self.labels = loadmat(self.PATH_POSE, struct_as_record = False, squeeze_me = True)
                self.labels = self.labels['RELEASE']

        def load_npy_images(self, part, data_type):
            """
            Function fetches and loads specified batch file with processed images. For loading respective labels
            see function load_npy_labels

            :param part: Number of batch to be loaded
            :param data_type: Either 'est' for estimation dataset or 'val' for validation dataset
            :return: Numpy array with processed images
            """
            name = '%s_part%02d.npy' % (data_type, part)
            path = os.path.join(self.NPY_PATH_IMAGES, name)

            return np.load(path)

        def load_npy_labels(self, data_type):
            """
            Function fetches and loads files with processed labels, visibility info and information whether they are
            present in image. For loading respective images see function load_npy_images

            :param data_type: Either 'est' for estimation dataset or 'val' for validation dataset
            :return: Numpy arrays with processed joint locations, visibility and presentation information
            """
            labels_path = os.path.join(self.NPY_PATH_POSE, '%s_labels.npy' % data_type)
            visible_joints_path = os.path.join(self.NPY_PATH_POSE, '%s_visible_joints.npy' % data_type)
            present_joints_path = os.path.join(self.NPY_PATH_POSE, '%s_present_joints.npy' % data_type)

            return np.load(labels_path), np.load(visible_joints_path), np.load(present_joints_path)

        def get_labels_in_batch(self, labels, batch_num):
            """
            Returns all labels that belonging to images in given number of batch

            :param labels: All labels to be divided. E.g. tuple: (joint positions, visibility, presentation). Can be
                           taken from output of methods load_npy_labels or get_cropped_labels
            :param batch_num: Number of batch
            :return: All labels that belonging to images in given number of batch
            """

            a = batch_num * self.NPY_BATCH_SIZE
            b = a + self.NPY_BATCH_SIZE
            b = min(b, labels[0].shape[0])

            return [x[a:b] for x in labels]

        def get_cropped_labels(self, cropped_size, dataset = None):
            """
            Returns labels, visibility and presentation information belonging to cropped frames - output of
            get_cropped_frames method.

            :param cropped_size: Size of cropped frames
            :param dataset: Set of image names. Frames containing image with name that is not presented in this set
                            will be ignored
            :return: Sequence of labels, visibility and presentation information belonging to cropped and scaled images
            """
            for pose_image in self.get_images():
                if dataset is not None and os.path.basename(pose_image.im_path) not in dataset:
                    continue

                labels = np.zeros((2, 16), dtype = np.int32) - 1
                is_visible = np.ones((16,), dtype = np.bool)
                is_present = np.zeros((16,), dtype = np.bool)

                for l in pose_image.get_joints_positions():
                    labels[:, l[0]] = pose_image.adjust_cropped_joint_position((l[2], l[3]), cropped_size)
                    is_visible[l[0]] = l[1] == 1
                    is_present[l[0]] = True

                yield labels, is_visible, is_present

        def get_cropped_frames(self, cropped_size, dataset = None):
            """
            Returns frames containing cropped and scaled image of given size, centered around person.

            :param cropped_size: Size of output frames
            :param dataset: Set of image names that have to be processed and returned. Frames containing image with
                            name that is not presented in this set will be ignored
            :return: Sequence of cropped and scaled frames centered around particular person
            """
            for pose_image in self.get_images():
                if dataset is not None and os.path.basename(pose_image.im_path) not in dataset:
                    continue

                X = pose_image.get_cropped_frame(cropped_size)

                yield X

        def get_images(self):
            """
            This function returns sequence of image information, including image path and label information
            """

            if self.labels is None:
                self.load()

            is_train = self.labels.img_train
            pose_info = self.labels.annolist

            for i, info in enumerate(pose_info):
                if is_train[i] == 0:
                    continue

                img_name = info.image.name
                img_path = os.path.join(self.PATH, img_name)
                positions = info.annorect

                if not isinstance(positions, np.ndarray):
                    positions = [positions]

                try:
                    for person in positions:
                        if isinstance(person.annopoints, np.ndarray) and person.annopoints.shape[0] == 0:
                            continue

                        joints = person.annopoints.point
                        position = person.objpos

                        if not isinstance(joints, np.ndarray):
                            continue

                        joints_info = sorted((joint.id, joint.is_visible, joint.y, joint.x)
                                             for joint in joints)

                        img = PoseImage(
                            im_path = img_path,
                            y = position.y,
                            x = position.x,
                            scale = person.scale,
                            joints = joints_info
                         )

                        yield img

                except AttributeError:
                    pass

    class _LSP(_PoseDatasetInfo):
        pass

    def __init__(self):
        self.gait_datasets = []  # type: List[List[Person]]

        # Load CASIA A, B and C datasets
        self.gait_casia_a = self._CasiaA()
        self.gait_casia_b = self._CasiaB()
        self.gait_casia_c = self._CasiaC()

        # Load Human 3.6M dataset
        self.pose_h36m = self._H36m()
        self.mpii = self._MPII()

        datasets_to_load = [
            self.gait_casia_a,
            self.gait_casia_b,
            self.gait_casia_c
        ]

        # self._load_gait_data(datasets_to_load)

    def _load_gait_data(self, datasets_to_load) -> None:
        self.gait_datasets = [list(dataset.get_persons()) for dataset in datasets_to_load]


class Sequence(object):

    def __init__(self, sequence_id: str, path: str, h: int, w: int):
        self.h = h
        self.w = w
        self.path = path
        self.sequence_id = sequence_id

    @abstractmethod
    def get_frames(self):
        pass


class GaitSequence(Sequence):
    """

        :param speed: Speed of walk can be one of these values: ['normal', 'fast', 'slow']
    """
    def __init__(self, sequence_id: str, path: str, background_path: str, h: int, w: int,
                 is_infrared: bool = False, speed: str = 'normal', has_bag: bool = False,
                 has_clothes: bool = False, angle: int = 90, has_shadow: bool = False, frame_clip = None):
        super().__init__(sequence_id, path, h, w)

        self.background_path = background_path
        self.is_infrared = is_infrared
        self.speed = speed
        self.has_bag = has_bag
        self.has_clothes = has_clothes
        self.angle = angle
        self.frame_clip = frame_clip
        self.has_shadow = has_shadow

    @abstractmethod
    def get_frames(self):
        pass

    @abstractmethod
    def get_background_frames(self):
        pass

    def get_background_frames_c(self, count):
        i = 1
        while True:
            for bg_frame in self.get_background_frames():
                yield bg_frame

                i += 1
                if i > count:
                    break
            else:
                continue
            break


class ImageGaitSequence(GaitSequence):

    def get_background_frames(self):
        yield load_image(self.background_path)

    def get_frames(self):
        for im_name in os.listdir(self.path):
            im_path = os.path.join(self.path, im_name)

            yield load_image(im_path)


class VideoGaitSequence(GaitSequence):

    def get_background_frames(self):
        return self.read_video(self.background_path)

    def get_frames(self):
        return self.read_video(self.path)

    def read_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)

        while True:
            stat, frame = cap.read()

            if not stat:
                break

            if self.frame_clip is None:
                yield frame
            else:
                c = self.frame_clip
                yield frame[c[0]:c[1], c[2]:c[3]]


class PoseSequence(Sequence):
    def __init__(self, sequence_id: str, path: str, h: int, w: int,
                 positions_2D_path: str, positions_3D_path: str, silhouettes_path: str, action: str):
        super().__init__(sequence_id, path, h, w)

        self.action = action
        self.positions_3D_path = positions_3D_path
        self.positions_2D_path = positions_2D_path
        self.silhouettes_path = silhouettes_path

    def get_frames(self):
        return self.read_video(self.path)

    def get_foreground_masks(self):
        return self.read_video(self.silhouettes_path)

    def read_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)

        while True:
            stat, frame = cap.read()

            if not stat:
                break

            yield frame


class PoseImage(object):

    def __init__(self, im_path: str, x: int, y: int, scale: float, joints):
        self.y = y
        self.x = x
        self.scale = scale
        self.joints = joints
        self.im_path = im_path

    def get_frame(self) -> np.ndarray:
        return load_image(self.im_path)

    def get_cropped_frame(self, size) -> np.ndarray:
        return self.crop(self.get_frame(), size = size)

    def get_persons_positions(self):
        return self.x, self.y, self.scale

    def get_joints_positions(self):
        return self.joints

    def crop(self, img, size):
        r = to_int(100 * self.scale)

        cropped = slice_pad(img, self.y - r, self.y + r, self.x - r, self.x + r)
        resized = imresize(cropped, (size, size))

        return resized

    def adjust_cropped_joint_position(self, joint, size):
        r = 100 * self.scale

        y = joint[0] - (self.y - to_int(r))
        x = joint[1] - (self.x - to_int(r))

        w = size / (2*r)

        y = to_int(w * y)
        x = to_int(w * x)

        return [y, x]

class Person:
    """
        Class stores basic information of the person to be identified,
        e.g. path to frames from video sequences of their walk

        :param sequences:
            The list contains all walk sequences related to this person
        :type sequences: List[GaitSequence]
    """

    def __init__(self, person_id: str, sequences = None):
        if sequences is None:
            sequences = []

        self.person_id = person_id
        self.sequences = sequences

    def add_sequence(self, val: GaitSequence):
        """ Add new walk sequence of this person """

        self.sequences.append(val)

    def __str__(self):
        return "Person '" + self.person_id + "'"
