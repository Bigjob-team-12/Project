import sys
# path 설정
sys.path.append('C:/Users/kdan/BigJob12/main_project/_src/data_analysis/dog_image_similarity')
sys.path.append('C:/Users/kdan/BigJob12/main_project/_src/data_analysis/re_id/code')
import extract_similar_image_path, copy_image, predict_dog_data
import reid_query, flush_gpu
from numba import cuda
import tensorflow as tf



def main(location, date, model):

    # input image에 대해 공고 이미지 filtering
    print('call_1')

    # if first: model = predict_dog_data.make_model(256)

    extract_similar_image_path.main(location, date, model)
    print('call_2')

    # filtering된 image re_id 사용할 directory로 copy
    copy_image.main()
    print('call_3')

    reid_query.main()

    return 0