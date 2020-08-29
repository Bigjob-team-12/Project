import numpy as np
from PIL import Image
import cv2
import os
import pandas as pd
import sys
import gc
import torch
from predict_dog_data import get_steps, make_generators, make_model, make_predictions
from dog_breed_similarity_comparison import load_data, cos_sim, euc_sim, pearson
from numba import cuda
import pymysql
import csv
import pickle

# DB connect
conn = pymysql.connect(host='localhost', user='root', password='bigjob12',
                   db='project', charset='utf8')

province = {
    '서울': ['서울', '인천', '경기'],
    '인천': ['인천', '서울', '경기'],
    '대전': ['대전', '세종', '충북', '충남'],
    '대구': ['대구', '경북', '경남'],
    '울산': ['울산', '부산', '경북', '경남'],
    '부산': ['부산', '울산', '경남'],
    '광주': ['광주', '전남'],
    '세종': ['세종', '대전', '충북', '충남'],
    '경기': ['서울', '인천', '강원', '충북', '충남'],
    '강원': ['강원', '경기', '충북', '경북'],
    '충북': ['충북', '대전', '세종', '경기', '강원', '충남', '경북', '전북'],
    '충남': ['충남', '대전', '세종', '경기', '충북', '전북'],
    '경북': ['경북', '대구', '울산', '강원', '충북', '경남', '전북'],
    '경남': ['경남', '대구', '울산', '부산', '경북', '전북', '전남'],
    '전북': ['전북', '충북', '충남', '경북', '경남', '전남'],
    '전남': ['전남', '광주', '경남', '전북'],
    '제주': ['제주']
    }

def get_data_sets(dir, image_size):
    '''
    load image data
    :param dir: input image directory
    :param image_size:
    :return: image, file name
    '''
    data = []
    files = []
    dir_lists = os.listdir(dir)

    for image_dir in dir_lists:
        file_path = os.path.join(dir, image_dir)
        if os.path.isfile(file_path):
            # 한글 directory 오류 방지
            print(file_path)
            ff = np.fromfile(file_path, np.uint8)
            img = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED)
            image_array = Image.fromarray(img, 'RGB')
            data.append(np.array(image_array.resize((image_size, image_size))))

            files.append(image_dir)

    return np.array(data), np.array(files)
def compare_similarities_and_show_results(predict, location, date, sim_func = pearson, n=10):
    '''
    유사도 비교 후 높은 순으로 10개 보여주기
    :param predict: softmax 확률값
    :param image_path:
    :param n: 보여줄 image 갯수
    :return: None
    '''
    # data = load_data().reset_index()
    # data = data[data['file_name'].apply(lambda x : True if location in x else False)].set_index('file_name')

    data = load_data()

    # print(data.head())
    print(location, date.replace('-',''))

    # 날짜 filtering
    date = int(date.replace('-',''))

    print(date)

    print('raw data')
    print(data.shape)
    data = data[data.start.apply(lambda x : date < x)]
    # data = data[data.end.apply(lambda x: date < x + 10)]

    print()
    print('date filtering')
    print(data.shape)

    # 지역 filtering
    data = data[data['name'].apply(lambda x : x[:2] in province[location])]
    # data = data[data['name'].apply(lambda x : x[:2] == location)]
    print()
    print('지역 filtering')
    print(data.shape)

    data = data.iloc[:, :30]

    # file_list = data.apply(lambda x: sim_func(x, predict[0]), axis=1).sort_values(ascending=False).index[:n]
    new_data = data.apply(lambda x: sim_func(x, predict[0]), axis=1).sort_values(ascending=False)
    new_data = new_data[new_data.apply(lambda x : x[0] > 0)]

    print()
    print('유사도 filtering')
    print(new_data.shape)

    return new_data
def show_similar_images(source_dir,output_dir,location, date, model, image_size=224,rand_seed=128):
    '''
    입력한 이미지와 저장되어 있는 공고 데이터와의 유사도 비교 후 10개 보여주기
    :param source_dir: input image directory
    :param output_dir: model and softmax data directory
    :param image_path: 저장되어 있는 image directory
    :param image_size:
    :param rand_seed:
    :return: None
    '''
    # load one image
    print('similar1')
    print(source_dir)
    test_data, test_files = get_data_sets(source_dir,image_size)
    t_steps, t_batch_size = get_steps(test_data)
    test_gen = make_generators(test_data, t_batch_size)
    # print(np.array(test_gen))
    print('-' * 30)
    # [print(_) for _ in test_gen]
    #
    # # print(test_gen.__next__())
    # print('-' * 30)
    # print(test_data)
    print('similar2')
    # init model
    # model = make_model(rand_seed)
    # predcit
    print('similar3')
    predict = make_predictions(test_gen, t_steps, model)


   # predict = make_predictions(test_data,  model)

    # 유사한 image 보여주기
    file_path = compare_similarities_and_show_results(predict, location, date, sim_func = pearson)

    return file_path
def main(location, date, model):
    #source_dir = 'C:/Users/kdan/BigJob12/main_project/_src/web/static/images/input_image'  # 쿼리 이미지
    source_dir ='C:/Users/kdan/BigJob12/main_project/_db/data/model_data/query/query_list'
    output_dir = 'C:/Users/kdan/BigJob12/main_project/_db/data/model_data/working'  # 모델
    image_path = 'C:/Users/kdan/BigJob12/main_project/_db/data/Preprocessed_data/'  # 공고 이미지
    # image_path = '../../../_db/data/input_query/input/dog_data/ours_dog/test'
    # location = '경북'

    data = [location, date]

    # with open(output_dir + '/' + 'data.pickle', 'wb') as f:
    #     pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    with open(output_dir + '/' + 'data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

    image_size = 224
    rand_seed = 256
    print('extract1')
    file_path = show_similar_images(source_dir, output_dir, location, date, model, image_size=image_size,
                                    rand_seed=rand_seed)
    print('extract2')

    pd.DataFrame(file_path).to_csv(output_dir + '/to_reid.csv', encoding='u8')


if __name__ == '__main__':
    main()