import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from PIL import Image
import cv2
import os
from tqdm import tqdm
import pymysql

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image



# DB connect
conn = pymysql.connect(host='localhost', user='root', password='bigjob12',
                   db='project', charset='utf8')

def get_data_sets(test_path, image_size):
    '''
    load test data
    :param dir: test dataset directory
    :param image_size:
    :return: test data, label, file name, class list
    '''
    # save class list
    class_list = [s for s in os.listdir(test_path)]

    data = []
    labels = []
    files = []

    # load image file
    class_count = -1
    for _class in class_list:
        d_path = os.path.join(test_path, _class)  # path to class directories
        if os.path.isdir(d_path):
            class_count = class_count + 1
            file_list = os.listdir(d_path)

            # tqdm / file 진행 확인용 변수
            d_list = d_path.split(r'/')
            length = len(d_list)
            desc = d_list[length - 2] + '-' + d_list[length - 1]

            for f in tqdm(file_list, desc=desc, unit='files', leave=True):
                file_path = os.path.join(d_path, f)

                # 한글 경로 인식 문제
                ff = np.fromfile(file_path, np.uint8)
                img = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED)

                try:
                    image_array = Image.fromarray(img, 'RGB')
                except:
                    print(file_path)
                    continue

                data.append(np.array(image_array.resize((image_size, image_size))))
                labels.append(class_count)
                files.append(f)

    return np.array(data), np.array(labels), np.array(files), class_list
def print_data(test_labels, class_list):
    '''
    print number of data per class
    :param test_labels: test label
    :param class_list: class list
    :return: None
    '''
    test_list=list(test_labels)
    print('{0:9s}Class Name{0:10s}Class No.{0:4s}Test Files{0:5s}'.format(' '))

    for i in range(0, len(class_list)):
        c_name=class_list[i]
        tf_count=test_list.count(i)
        print('{0}{1:^25s}{0:5s}{2:3.0f}{0:9s}{3:4.0f}'.format(' ',c_name,i,tf_count))

    print('{0:30s} ______________________________________________________'.format(' '))
    msg='{0:10s}{1:6s}{0:16s}{2:^3.0f}{0:8s}{3:3.0f}\n'
    print(msg.format(' ', 'Totals',len(class_list),len(test_labels)))
def get_steps(test_data):
    '''
    step and batch size
    :param test_data:
    :return: step, batch size
    '''
    length=test_data.shape[0]

    batches=[int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=80]
    batches.sort(reverse=True)
    t_batch_size=batches[0]
    t_steps=length/t_batch_size
    return t_steps, t_batch_size
def make_generators(test_data, t_batch_size, test_labels=None):
    '''
    image preprocessing
    :param test_data:
    :param test_labels:
    :param t_batch_size:
    :return: preprocessed data
    '''
    test_datagen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,
                                    samplewise_center=True,
                                    samplewise_std_normalization=True)
    test_gen=test_datagen.flow(test_data, y=test_labels, batch_size=t_batch_size, shuffle=False)

    return test_gen
def make_model(rand_seed, output_dir = 'C:/Users/kdan/BigJob12/main_project/_db/data/model_data/working', size=30):
    '''
    make initial model
    :param rand_seed:
    :param size: class size
    :return: model
    '''
    print('make_model1')
    tf.keras.backend.clear_session()
    # mobile = mobile
    mobile = tf.keras.applications.MobileNetV2()

    # remove last 5 layers of model and add dense layer with 128 nodes and the prediction layer with size nodes
    # where size=number of classes
    x = mobile.layers[-2].output
    x = Dense(128, kernel_regularizer=regularizers.l2(l=0.015), activation='relu')(x)
    x = Dropout(rate=.5, seed=rand_seed)(x)
    predictions = Dense(size, activation='softmax')(x)
    model = Model(inputs=mobile.input, outputs=predictions)
    for layer in model.layers:
        layer.trainable = True

    model.compile(Adam(lr=0.0015), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model_path = os.path.join(output_dir, 'tmp.h5')

    model.load_weights(model_path)

    return model

def make_predictions(test_gen, t_steps, model):
    '''
    품종별 분류 될 확률
    :param output_dir: model directory
    :param test_gen:
    :param t_steps:
    :param model: init model
    :return: predict
    '''
    test_gen.reset()
    pred = model.predict(test_gen)

    return pred
def save_predicted_value_as_csv(predict, test_labels, test_files, class_list, output_dir):
    '''
    save predict
    :param predict:
    :param test_labels:
    :param test_files:
    :param class_list:
    :return: None
    '''
    curs = conn.cursor()
    sql = 'SELECT number, deadline from protect_animals_url1'
    curs.execute(sql)
    rows = curs.fetchall()

    rows = dict(rows)

    result = pd.DataFrame(predict)

    result['true_class'] = test_labels
    result['predict_class'] = result.iloc[:, :-1].idxmax(axis=1)
    result['class_name'] = list(map(lambda x: class_list[x], result['true_class']))
    result['file_name'] = result['class_name'] + '/' + test_files
    result['name'] = result['file_name'].apply(lambda x: x.split('_')[-1][:-4])
    result['deadline'] = result['name'].apply(lambda x: rows[x])
    result['start'] = result['deadline'].apply(lambda x : x.split()[0])
    result['end'] = result['deadline'].apply(lambda x : x.split()[-1])

    print(result.shape)

    tmp_result = pd.read_csv(output_dir + '/result.csv').iloc[:, 1:]

    result.columns = tmp_result.columns

    print(tmp_result.shape)

    tmp_result = pd.concat([result, tmp_result], ignore_index=True)

    print(tmp_result.shape)

    tmp_result.to_csv(output_dir + '/result.csv')
def TF2_classify(source_dir,output_dir,image_size=224,rand_seed=128):
    '''
    classification by breeds and save predict
    :param source_dir: test dataset directory
    :param output_dir: model directory and save predict
    :param image_size:
    :param rand_seed:
    :return: None
    '''
    # load test data
    test_data, test_labels, test_files, class_list = get_data_sets(source_dir,image_size)

    t_steps, t_batch_size = get_steps(test_data)
    test_gen = make_generators(test_data, t_batch_size)

    # init model
    model = make_model(rand_seed)
    # predict
    predict = make_predictions(test_gen, t_steps, model)
    save_predicted_value_as_csv(predict, test_labels, test_files, class_list, output_dir)

if __name__ == '__main__':
    # test data directory
    source_dir = '../../../_db/data/Crawling_data/[개]'
    # model directory
    output_dir='../../../_db/data/model_data/working'
    image_size=224
    rand_seed=256

    TF2_classify(source_dir,output_dir,image_size=image_size,rand_seed=rand_seed)