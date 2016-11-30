from time import time

_time_now = time()


def tic():
    global _time_now
    _time_now = time()


def toc():
    print('time: ' + str(time() - _time_now) + 's')
