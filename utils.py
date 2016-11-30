from functools import lru_cache
import math

import numpy as np
from scipy.stats import multivariate_normal


def slice_pad(arr: np.ndarray, y1: int, y2: int, x1: int, x2: int) -> np.ndarray:
    shape = arr.shape
    h, w = shape[0:2]
    ch = 1

    if len(shape) == 3:
        ch = shape[2]

    if x1 >= 0 and x2 <= w and y1 >= 0 and y2 <= h:
        if ch > 1:
            return arr[y1:y2, x1:x2, :]
        else:
            return arr[y1:y2, x1:x2]

    yd = y2 - y1
    xd = x2 - x1
    res = np.zeros((yd, xd), dtype = arr.dtype) if ch == 1 \
        else np.zeros((yd, xd, ch), dtype = arr.dtype)

    y1fr = max(0, y1)
    y2fr = min(y2, h)
    x1fr = max(0, x1)
    x2fr = min(x2, w)

    y1to = max(0, yd - y2)
    y2to = y1to + y2fr - y1fr

    x1to = max(0, xd - x2)
    x2to = x1to + x2fr - x1fr

    if ch > 1:
        for i in range(ch):
            res[y1to:y2to, x1to:x2to, i] = arr[y1fr:y2fr, x1fr:x2fr, i]
    else:
        res[y1to:y2to, x1to:x2to] = arr[y1fr:y2fr, x1fr:x2fr]

    return res


def get_affine_transform(v, T):
    nv = np.ones((v.shape[0] + 1, v.shape[1]), dtype = np.float32)
    nv[0:2, :] = v
    return np.array(np.round(np.matmul(T, nv)), dtype = np.int32)[0:2, :]


@lru_cache()
def get_gauss_pdf(sigma, is_visible):
    n = sigma * 8

    x, y = np.mgrid[0:n, 0:n]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    rv = multivariate_normal([n / 2, n / 2], [[sigma ** 2, 0], [0, sigma ** 2]])
    pdf = rv.pdf(pos)

    heatmap = pdf / np.max(pdf)

    # if not is_visible:
    #     heatmap *= 0.5

    return heatmap


def take_slices(it, n):
    arr = []

    for i, s in enumerate(it):
        arr.append(s)

        if (i + 1) % n == 0:
            yield arr
            arr = []

    if len(arr) != 0:
        yield arr


def to_int(num):
    return int(round(num))


@lru_cache()
def get_binary_mask(diameter):
    d = diameter
    map = np.zeros((d,d), dtype = np.float32)

    r = d/2
    s = int(d/2)

    y, x = np.ogrid[-s:d - s, -s:d - s]
    mask = x * x + y * y <= r * r

    map[mask] = 1.0

    return map


def get_binary_heat_map(shape, is_present, centers, diameter = 9):
    n = diameter
    hn = int(n / 2)
    pl = np.zeros((shape[0], shape[1] + n + n, shape[2] + n + n, shape[3]), dtype = np.float32)

    for i in range(shape[0]):
        for j in range(shape[3]):
            my = centers[i, 0, j] - hn
            mx = centers[i, 1, j] - hn

            if -hn < my < shape[1] and -hn < mx < shape[2] and is_present[i, j]:
                pl[i, my + n:my + n + n, mx + n:mx + n + n, j] = get_binary_mask(diameter)

    return pl[:, n:-n, n:-n, :]


def get_gauss_heat_map(shape, is_visible, mean, sigma = 5):
    n = to_int(sigma * 8)
    hn = to_int(n / 2)
    pl = np.zeros((shape[0], shape[1] + n + n, shape[2] + n + n, shape[3]), dtype = np.float32)

    for i in range(shape[0]):
        for j in range(shape[3]):
            my = mean[i, 0, j] - hn
            mx = mean[i, 1, j] - hn

            if 0 < my < shape[1] and 0 < mx < shape[2]:
                pl[i, my + n:my + n + n, mx + n:mx + n + n, j] = get_gauss_pdf(sigma, is_visible[i, j])
                # else:
                #     print(my, mx)

    return pl[:, n:-n, n:-n, :]


def take_slice(arr, i, size):
    a = i * size
    b = a + size
    b = min(b, arr.shape[0])

    return arr[a:b, ...]


def take_n(arr, n):
    for i, item in enumerate(arr):
        if i >= n: break

        yield item


def savitzky_golay(y, window_size, order, deriv = 0, rate = 1):
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2

    # pre-compute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)

    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode = 'valid')
