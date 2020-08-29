import pymysql
import urllib.request
import os
import socket
import sys
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
# path 설정
# sys.path.append('C:/Users/kdan/BigJob12/main_project/_src/data_processing/yolo_v4')
# import detect

# set download timeout
socket.setdefaulttimeout(30)

# DB connect
conn = pymysql.connect(host='localhost', user='root', password='bigjob12',
                       db='project', charset='utf8')
# #YOLO weights 경로
# weights = 'C:/Users/kdan/BigJob12/main_project/_src/data_processing/yolo_v4/checkpoints/yolov4-416'
# saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])


def load_image():
    '''
    crawling 된 데이터 load
    :return: crawling data
    '''
    curs = conn.cursor()
    # 최근 crawling된 data부터 load
    sql = 'SELECT distinct image, number, kind from protect_animals_url1 ORDER BY NO desc'
    curs.execute(sql)
    images = curs.fetchall()
    return images
def download_image(images):
    '''
    crawling된 data download
    :param images: crawling data(image url, number, kind)
    :return: None
    '''
    download_path = '../../_db/data/Crawling_data/[개]/'

    image_path = '../../_db/data/Preprocessed_data/'

    for url, number, class_name in images:
        dog_cat, *class_name = class_name.split()

        if dog_cat == '[개]': pass
        else: continue

        tmp_path = download_path

        class_name = '_'.join(class_name) if len(class_name) else 'none'

        tmp_path = tmp_path + '/' + class_name
        if not os.path.isdir(tmp_path):
            os.mkdir(tmp_path)
        if not os.path.isdir(image_path + '/' + class_name):
            os.mkdir(image_path + '/' + class_name)

        try:
            # download image
            print(class_name + '_' + number)
            # file 존재할 경우 break
            if os.path.isfile(image_path + '/' + class_name + '/' + class_name + '_' + number + ".jpg"):
                print('해당 파일 존재')
                break
            else:
                urllib.request.urlretrieve(url, tmp_path + '/' + class_name + '_' + number + ".jpg")
                # img_path= tmp_path + '/' + class_name + '_' + number + ".jpg"
                # detect.main(img_path, saved_model_loaded)
        except:
            # 해당 file remove
            if os.path.isfile(tmp_path + '/' + class_name + '_' + number + ".jpg"):
                os.remove(tmp_path + '/' + class_name + '_' + number + ".jpg")
            print('download error : ' + number)

if __name__ == '__main__':
    # load image url, name
    images = load_image()

    # download image
    download_image(images)
