import sys
# path 설정
sys.path.append('C:/Users/kdan/BigJob12/main_project/_src/data_analysis/dog_image_similarity')
sys.path.append('C:/Users/kdan/BigJob12/main_project/_src/data_analysis/re_id/code')

import extract_similar_image_path, copy_image
import reid_test# , flush_gpu
from numba import cuda



def main():
    # input image에 대해 공고 이미지 filtering
    extract_similar_image_path.main()

    # filtering된 image re_id 사용할 directory로 copy
    copy_image.main()

    # re_id를 이용한 유사한 이미지 추출
    device = cuda.get_current_device()
    device.reset()
    reid_test.main()
