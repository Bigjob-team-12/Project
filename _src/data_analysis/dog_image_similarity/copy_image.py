import shutil
import pandas as pd
import os

if __name__ == '__main__':
    data = pd.read_csv('test.csv')

    path = '../../../_db/data/model_data/gallery/'

    if not os.path.isdir(path):
        os.mkdir(path)

    shutil.rmtree(r'C:\Users\user\PycharmProjects\Project\_db\data\model_data\gallery')

    if not os.path.isdir(path):
        os.mkdir(path)

    image_path = '../../../_db/data/Crawling_data/[개]/'

    for image_file_path in data['file_name']:
        shutil.copy(image_path + image_file_path, path + image_file_path.split('/')[-1])

    print(len(data['file_name']))