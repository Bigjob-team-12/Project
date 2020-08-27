import reid_test
from numba import cuda

def main():
    device = cuda.get_current_device()
    device.reset()
    reid_test.main()
