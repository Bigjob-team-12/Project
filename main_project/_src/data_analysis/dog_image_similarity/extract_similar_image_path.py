import numpy as np
from PIL import Image
import cv2
import os
import pandas as pd
from predict_dog_data import get_steps, make_generators, make_model, make_predictions
from dog_breed_similarity_comparison import load_data, cos_sim, euc_sim, pearson
import pymysql
import csv

# DB connect
conn = pymysql.connect(host='localhost', user='root', password='bigjob12',
                   db='project', charset='utf8')
# 지역별 가까운 지역 추가
province = {
    '전국': ['서울', '인천', '경기','강원','울산', '부산', '경북', '경남','대전', '대구', '세종', '충북', '충남', '전남', '광주', '전북','제주'],
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
def compare_similarities_and_show_results(predict, location, date, sim_func = pearson):
    '''
    유사도 비교 후 높은 순으로 10개 보여주기
    :param predict: softmax 확률값
    :param location: 사용자 입력 : 지역
    :param date: 사용자 입력 : 날짜
    :return: filtered data
    '''
    data = load_data()

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
    print()
    print('지역 filtering')
    print(data.shape)

    data = data.iloc[:, :30]

    new_data = data.apply(lambda x: sim_func(x, predict[0]), axis=1).sort_values(ascending=False)
    new_data = new_data[new_data.apply(lambda x : x[0] > 0)]

    print()
    print('유사도 filtering')
    print(new_data.shape)

    return new_data
def show_similar_images(source_dir,location, date, model, image_size=224):
    '''
    입력한 이미지와 저장되어 있는 공고 데이터와의 유사도 비교
    :param source_dir: input image directory
    :param location: 사용자 입력 : 지역
    :param date: 사용자 입력 : 날짜
    :param model: 품종분류기 model
    :return: 유사한 이미지 file path
    '''
    # load one image
    test_data, test_files = get_data_sets(source_dir,image_size)
    t_steps, t_batch_size = get_steps(test_data)
    test_gen = make_generators(test_data, t_batch_size)

    # predcit
    predict = make_predictions(test_gen, t_steps, model)
    # 유사한 image 보여주기
    file_path = compare_similarities_and_show_results(predict, location, date, sim_func = pearson)

    return file_path
def main(location, date, model):
    '''
    location, date & softmax값 기준 피어슨 상관계수 이상인 image filtering
    :param location: 사용자 입력 : 지역
    :param date: 사용자 입력 : 날짜
    :param model: 품종분류기 model
    '''
    source_dir ='C:/Users/kdan/BigJob12/main_project/_db/data/model_data/query/query_list'
    output_dir = 'C:/Users/kdan/BigJob12/main_project/_db/data/model_data/working'  # 모델

    data = [location, date]

    with open(output_dir + '/' + 'data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

    image_size = 224
    file_path = show_similar_images(source_dir, location, date, model, image_size=image_size)

    pd.DataFrame(file_path).to_csv(output_dir + '/to_reid.csv', encoding='u8')


if __name__ == '__main__':
    main()