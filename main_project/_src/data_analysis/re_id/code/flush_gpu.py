import reid_query
from numba import cuda

def main():
    # device = cuda.get_current_device()
    # device.reset()
    reid_query.main()
