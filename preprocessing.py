# Standard libraries
import os
import typing
from fractions import Fraction
from itertools import zip_longest
from typing import Any, Iterable, List, Callable
from abc import abstractmethod

# Third party libraries
import cv2
import cv2.optflow as optf
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import time
from scipy.misc import imresize

# Local libraries
from scipy.ndimage import center_of_mass

import data
from data import GaitSequence
from utils import slice_pad


# extract rect
# def rect ratio + resize to def size


class SequenceProcessing(object):
    def __init__(self, sequence: GaitSequence):
        self.sequence = sequence
        self.labels = None

        self.MIN_AREA = 300
        self.RECTANGLE_SIZE = (100, 75)
        self.RECTANGLE_PADDING = (10, 10)
        self.RECTANGLE_RATIO = Fraction(*self.RECTANGLE_SIZE)

    def pre_process(self, frame: np.ndarray) -> np.ndarray:
        """
            Pre-process each frame of the sequence.

            :param frame: The frame to process
            :return: Processed frame - 3 channel or 1 channel (for infrared) np array [int32]
        """

        size = 5
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        # frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

        if self.sequence.is_infrared:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.equalizeHist(frame)

        # frame = np.array(frame * 1)
        frame = cv2.GaussianBlur(frame, (size, size), 2)

        return frame

    def get_bounding_box(self, fg_mask: np.ndarray) -> np.ndarray or None:
        """
            The method returns position of bounding box of pedestrian if present

            :param frame: Frame from witch the pedestrian has to be cropped
            :param fg_mask: Foreground mask containing silhouette
            :return: Coordinates of bounding box in ratio ~ 4:3 (100px x 75px) containing person. If the person is no
                     presented in the input frame or it touches the edges (it is not completely displayed) method will
                     return None
        """


        def get_bounding_box_position(stat: List[int]) -> np.ndarray:
            """
                This method cut the pedestrian off (connected component described with the stat property) into a proper
                rectangle with fixed size (given in variable RECTANGLE_SIZE in outer scope)

                :param stat: Descriptions of the connected component belonging (probably) to pedestrian
                :return: Segmented pedestrian, align in center according to its centroid
            """
            W = stat[cv2.CC_STAT_WIDTH] + 2 * self.RECTANGLE_PADDING[1]
            H = stat[cv2.CC_STAT_HEIGHT] + 2 * self.RECTANGLE_PADDING[0]

            ratio = Fraction(H, W)

            if self.RECTANGLE_RATIO < ratio:
                # Silhouette is thinner then default rectangle (It should happen in every case)
                top, bottom = stat[cv2.CC_STAT_TOP], stat[cv2.CC_STAT_TOP] + stat[cv2.CC_STAT_HEIGHT]
                top -= self.RECTANGLE_PADDING[0]
                bottom += self.RECTANGLE_PADDING[1]

                scaled_w = H / self.RECTANGLE_RATIO

                # centroid of silhouette
                center = tuple(center_of_mass(fg_mask))
                left = int(center[1]) - int(scaled_w / 2)
                right = left + int(scaled_w)

                return top, bottom, left, right

            else:
                # Silhouette is wider - it shouldn't happen (especially the case with big difference between ratios)
                raise Exception("Bad ratio" + self.RECTANGLE_RATIO + ' ' + ratio)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 15))
        contour = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)

        h, w = np.shape(contour)
        labels = np.zeros((h, w), dtype = np.int32)

        # Finding connected components, the small ones will be removed
        cc_stats = cv2.connectedComponentsWithStats(contour, labels, 4, cv2.CV_32S)

        if cc_stats[0] > 1:
            # Get True value if there is at least one white pixel on leftmost or rightmost side
            sides_borders = np.any((fg_mask[:, 0] != 0) | (fg_mask[:, w - 1] != 0))

            # Select only frames with appropriate area and that don't touch the edges
            cc = [(stat[cv2.CC_STAT_AREA], i, stat) for i, stat in enumerate(cc_stats[2])
                  if i > 0
                  and stat[cv2.CC_STAT_AREA] > self.MIN_AREA
                  and not sides_borders
                  or (stat[cv2.CC_STAT_LEFT] > 0
                      and stat[cv2.CC_STAT_LEFT] + stat[cv2.CC_STAT_WIDTH] < w)]

            # Select frame with the biggest area
            if len(cc) > 0:
                best = max(cc)
                idx = best[1]
                stat = best[2]

                # Interception with dilated components and original frame
                fg_mask[labels != idx] = 0

                return get_bounding_box_position(stat)

        return None

    def get_rescaled_crop_image(self, frame: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
        frame_cut = slice_pad(frame, top, bottom, left, right)

        frame_scaled = imresize(
            arr = frame_cut,
            size = self.RECTANGLE_SIZE,
            interp = 'cubic',
            mode = None)

        return frame_scaled

    def get_background_frames(self, count: int = None) -> Iterable[np.ndarray]:
        """
            Method returns background images of the sequence

            :param count: Number of background frames to be returned. If there is not enough
                          background frames then they will repeat again from start. If the count is not
                          given or is None, this method uses just available frames of background
        """
        bg_frames = self.sequence.get_background_frames() if count is None else \
                    self.sequence.get_background_frames_c(count)

        for bg_frame in bg_frames:
            yield self.pre_process(bg_frame)

    def get_processed_frames(self):

        bg_subtractor = KNNBackgroundSubtractor(
            sequence = self.sequence,
            bg_frames_f = self.get_background_frames
        )

        opt_flow_proc = FOpticalFlow()

        frames = self.sequence.get_frames()

        for frame in frames:
            frame = self.pre_process(frame)
            bw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if opt_flow_proc.last_frame is None:
                opt_flow_proc.set_last_frame(bw_frame)
                continue

            fg_mask = bg_subtractor.get_fg_mask(frame)

            bb_coord = self.get_bounding_box(
                fg_mask = fg_mask
            )

            if bb_coord is not None:
                opt_flow = opt_flow_proc.get_optical_flow(bw_frame)
                f = opt_flow[:,:,0]

                print(np.max(f))

                yield self.get_rescaled_crop_image(frame, *bb_coord), frame
            else:
                opt_flow_proc.set_last_frame(bw_frame)



class OpticalFlow(object):

    def __init__(self):
        self.last_frame = None

    def set_last_frame(self, frame: np.ndarray) -> None:
        self.last_frame = frame

    def get_optical_flow(self, frame: np.ndarray) -> np.ndarray:
        flow = self.compute_optical_flow(frame)
        self.last_frame = frame

        return flow

    @abstractmethod
    def compute_optical_flow(self, frame: np.ndarray):
        pass


class FOpticalFlow(OpticalFlow):

    def compute_optical_flow(self, frame: np.ndarray) -> np.ndarray:

        return cv2.calcOpticalFlowFarneback(frame, self.last_frame,
                                            flow = None,
                                            pyr_scale = 0.8,
                                            levels = 5,
                                            winsize = 10,
                                            iterations = 2,
                                            poly_n = 7,
                                            poly_sigma = 1.5,
                                            flags = 0
                  )

"""
class SFOpticalFlow(OpticalFlow):

    def compute_optical_flow(self, frame: np.ndarray) -> np.ndarray:

        return optf.calcOpticalFlowSF(frame, self.last_frame,
                      layers = 3,
                      averaging_block_size = 2,
                      max_flow = 4
                  )


class TVOpticalFlow(OpticalFlow):
    def __init__(self):
        super().__init__()
        self.of_proc = cv2.createOptFlow_DualTVL1()

    def compute_optical_flow(self, frame: np.ndarray) -> np.ndarray:
        h, w, c = frame.shape
        flow = np.zeros((*frame.shape[0:2],2), dtype = np.float32)
        self.last_frame = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return self.of_proc.calc(frame, self.last_frame, flow)


class DFOpticalFlow(OpticalFlow):

    def __init__(self):
        super().__init__()
        self.of_proc = optf.createOptFlow_DeepFlow()

    def compute_optical_flow(self, frame: np.ndarray) -> np.ndarray:
        h,w,c = frame.shape
        flow = np.zeros(frame.shape, dtype = np.uint8)
        self.last_frame = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        print(frame.dtype)
        return self.of_proc.calc(frame, self.last_frame, flow)
"""

class BackgroundSubtractor(object):

    def __init__(self, sequence: GaitSequence, bg_frames_f: Callable[[int], Iterable[np.ndarray]]):
        """

            :param sequence: Data sequence
            :param bg_frames_f: Function returning 'n' background frames to be used for initialization
        """

        self.sequence = sequence
        self.bg_frames_f = bg_frames_f
        self.bg_pro = None
        self.labels = None

        self.model_new_bg_subtractor()

    def pre_process(self, frame: np.ndarray) -> np.ndarray:
        """
            Pre-process frame before background modeling and foreground extraction.

            :param frame: The frame to process
            :return: Processed frame - 3 channel or 1 channel (for infrared) np array [int32]
        """

        size = 5

        if self.sequence.is_infrared:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.equalizeHist(frame)

        frame = cv2.GaussianBlur(frame, (size, size), 2)

        return frame

    def compute_fg_mask(self, frame: np.ndarray,
                        learning_rate: float = -1,
                        min_cc_area: int = 100,
                        pre_process_image: bool = False) -> np.ndarray:
        """
            This method reads frame and using the background subtraction object 'self.bg_pro'
            creates binary foreground mask (result depends on previously frames).
            After that, some post-process (mostly morphological) operations are performed.

            :param frame: Frame to process
            :param learning_rate: The learning rate in interval [0,1]
            :param min_cc_area: Minimal size of each connected component. After creating silhouette with connected
                                components of insufficient size, these small components will be assumed as a noise
                                and will
                                be removed.
            :param pre_process_image: If True, frame will be pre-processed with method 'pre_process()'
            :return: Foreground mask [uint8]
        """
        M_OPENING = (4, 4)
        M_CLOSING = (5, 10)

        h, w = self.sequence.h, self.sequence.w
        if self.labels is None:
            self.labels = np.zeros((h, w), dtype = np.int32)

        def post_process(img: np.ndarray) -> np.ndarray:
            # Remove shadows (in case the flag is present for this dataset)
            if self.sequence.has_shadow:
                img[img < 255] = 0

            # Apply morphology closing & opening
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, M_OPENING)
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, M_CLOSING)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_close)
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_open)

            # Finding connected components, the small ones will be removed
            cc_stats = cv2.connectedComponentsWithStats(img, self.labels, 4, cv2.CV_32S)

            if cc_stats[0] > 1:
                not_small_cc = [i for i, stat in enumerate(cc_stats[2])
                                if stat[cv2.CC_STAT_AREA] > min_cc_area and i > 0]

                mask = np.reshape(np.array(np.in1d(self.labels, not_small_cc)), (h, w))
                self.labels[mask] = 255

            return np.array(self.labels, dtype = np.uint8)


        # Optional pre-process of frame
        if pre_process_image:
            frame = self.pre_process(frame)

        # Creating foreground mask, using background subtractor 'bg_pro'
        silhouette = self.bg_pro.apply(image = frame, learningRate = learning_rate)

        # Post-process the foreground mask using morphology and other operations
        silhouette = post_process(silhouette)

        return silhouette

    def model_background(self, learning_rate: float = -1, count: int = None) -> None:
        """
            This method model variations of background. Its use is optional but in case of use it
            should be used before creating silhouettes

            :param learning_rate: The learning rate in interval [0,1]
            :param count: Number of background frames to be returned. If there is not enough
                          background frames then they will repeat again from start. If the count is not
                          given or is None, this method uses just available frames of background
        """

        for frame in self.bg_frames_f(count):
            self.bg_pro.apply(
                image = frame,
                learningRate = learning_rate
            )

    @abstractmethod
    def model_new_bg_subtractor(self) -> None:
        """
            Method has to initialize background subtractor and assign it to 'self.bg_pro' field. On the end,
            'self.model_background()' method with appropriate parameters must be called.
        """
        pass

    @abstractmethod
    def get_fg_mask(self, frame: np.ndarray, pre_process_image: bool = False) -> np.ndarray:
        """
            Compute a foreground mask from frame. Method 'self.compute_fg_mask()' should be used.

            :param pre_process_image: If True, frame will be pre-processed with method 'pre_process()'
            :param frame: Frame to process
            :return: Foreground mask of frame
        """
        pass


class KNNBackgroundSubtractor(BackgroundSubtractor):

    def get_fg_mask(self, frame: np.ndarray, pre_process_image: bool = False) -> np.ndarray:
        return self.compute_fg_mask(
            frame = frame,
            learning_rate = 0.001,
            min_cc_area = 20,
            pre_process_image = pre_process_image
        )

    def model_new_bg_subtractor(self) -> None:
        self.bg_pro = cv2.createBackgroundSubtractorKNN(
            history = 10,
            detectShadows = self.sequence.has_shadow,
            dist2Threshold = 1700)

        self.bg_pro.setNSamples(9)
        self.bg_pro.setkNNSamples(5)

        self.model_background(
            count = 10,
            learning_rate = 0.5
        )

# ---- BG subtractors ---- #

'''
def BG_MOG2(sq: Sequence):
    """
        After good parameter tuning has this method very good performance and is the fastest
        compared to the other methods.
    """
    frames_history = 5
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history = frames_history,
        detectShadows = sq.has_shadow,
        varThreshold = 200)

    subtractor.setComplexityReductionThreshold(0.5)

    pr = SequenceProcessing(sq, subtractor)
    pr.model_background(
        count = frames_history,
        learning_rate = 0.1)

    return pr.create_silhouettes(
        learning_rate = 0.01,
        min_cc_area = 30)


def BG_KNN(sq: Sequence):
    """
        The method is slow but has very good results
    """
    subtractor = cv2.createBackgroundSubtractorKNN(
        history = 10,
        detectShadows = sq.has_shadow,
        dist2Threshold = 1700)

    subtractor.setNSamples(9)
    subtractor.setkNNSamples(5)

    pr = SequenceProcessing(sq, subtractor)
    pr.model_background(
        count = 10,
        learning_rate = 0.5)

    return pr.create_silhouettes(
        learning_rate = 0.001,
        min_cc_area = 20)


def BG_MOG(sq: Sequence) -> Iterable[np.ndarray]:
    """
        This method offer quite good results it is comparable with KNN and MOG2 but
        is much slower compared with MOG2
    """
    subtractor = cv2.bgsegm.createBackgroundSubtractorMOG(
        history = 100,
        backgroundRatio = 0.8,
        noiseSigma = 9)

    pr = SequenceProcessing(sq, subtractor)
    pr.model_background(
        count = 100,
        learning_rate = 0.8)

    return pr.create_silhouettes(
        learning_rate = 0.001,
        min_cc_area = 100)


def BG_GMG(sq: Sequence):
    """
        Quite complex method, but for many tested parameters it reaches very poor performance
    """
    subtractor = cv2.bgsegm.createBackgroundSubtractorGMG(
        initializationFrames = 100,
        decisionThreshold = 0.5)

    subtractor.setMaxFeatures(7)
    subtractor.setQuantizationLevels(5)
    subtractor.setSmoothingRadius(0)
    subtractor.setBackgroundPrior(0.99)

    pr = SequenceProcessing(sq, subtractor)
    pr.model_background(
        count = 100,
        learning_rate = 0.0001)

    return pr.create_silhouettes(
        learning_rate = 0.001,
        min_cc_area = 300)


# ---- BG-INFRA subtractors ---- #

def BG_INFRA_GMG(sq: Sequence):
    frames_history = 100
    subtractor = cv2.bgsegm.createBackgroundSubtractorGMG(
        initializationFrames = frames_history,
        decisionThreshold = 0.9)

    subtractor.setMaxFeatures(13)
    subtractor.setQuantizationLevels(20)
    subtractor.setSmoothingRadius(5)
    subtractor.setBackgroundPrior(0.8)

    pr = SequenceProcessing(sq, subtractor)
    pr.model_background(
        count = frames_history,
        learning_rate = 0.005)

    return pr.create_silhouettes(
        learning_rate = 0.01,
        min_cc_area = 100)


def BG_INFRA_MOG(sq: Sequence):
    subtractor = cv2.bgsegm.createBackgroundSubtractorMOG(
        history = 200,
        backgroundRatio = 0.5,
        noiseSigma = 3)

    pr = SequenceProcessing(sq, subtractor)
    pr.model_background(
        count = 30,
        learning_rate = 0.3)

    return pr.create_silhouettes(
        learning_rate = 0.1,
        min_cc_area = 30)


def BG_INFRA_KNN(sq: Sequence):
    frames_history = 50
    subtractor = cv2.createBackgroundSubtractorKNN(
        history = frames_history,
        detectShadows = False,
        dist2Threshold = 150)

    subtractor.setNSamples(15)
    subtractor.setkNNSamples(4)

    pr = SequenceProcessing(sq, subtractor)
    pr.model_background(
        count = frames_history,
        learning_rate = 0.1)

    return pr.create_silhouettes(
        learning_rate = 0.1,
        min_cc_area = 50)


def BG_INFRA_MOG2(sq: Sequence):
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history = 100,
        detectShadows = False,
        varThreshold = 30)

    subtractor.setComplexityReductionThreshold(0.005)

    pr = SequenceProcessing(sq, subtractor)
    pr.model_background(learning_rate = 0.05)

    return pr.create_silhouettes(
        learning_rate = 0.005,
        min_cc_area = 30)
'''