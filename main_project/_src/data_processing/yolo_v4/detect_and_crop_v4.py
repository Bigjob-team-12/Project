import detect
import os
import time
import tensorflow as tf

from tensorflow.python.saved_model import tag_constants

# main code - 이미지를 새로생성
# 대상 이미지 class가 모여있는 폴더를 path_file에 입력.
path_file = './' + 'data/'+ 'post_dog'
weights = './checkpoints/yolov4-416'

saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])

for count, foldername in enumerate(os.listdir(path_file)):
    print(count, foldername)
    for count, filename in enumerate(os.listdir(path_file+ '/' +foldername)):
        print(count, filename)
        path_src_file = path_file+'/'+foldername + '/'+filename

#        dst_name = filename.split('/')[-1].split('.jpg')[0] +'_crop'+'.jpg'
#        path_dst_file =  path_file +'/'+foldername + '/'+ dst_name
        print(path_src_file)
        start = time.time()

        detect.main(path_src_file,saved_model_loaded)
        print('total time = ', time.time() - start)
