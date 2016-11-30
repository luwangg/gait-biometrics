import PIL.Image as Img
import numpy as np


def load_image(file_name):
    img = Img.open(file_name)
    # img.load()
    data = np.asarray(img)

    return data
