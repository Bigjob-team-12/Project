import shutil
import pandas as pd
import time

if __name__ == '__main__':
    data = pd.read_csv('test.csv')

    image_path = '../../../_db/data/crawling_data/post_dog(size_240)/'

    copy_path = '../../../_db/data/model_data/pet_re_id_data/'

    start = time.time()  # 시작 시간 저장

    for image_file_path in data['file_name']:
        try:
            shutil.copy(image_path + image_file_path, copy_path + image_file_path.split('/')[-1])
        except:
            pass

    print(len(data['file_name']))
    print("time :", time.time() - start)

    # shutil.copy(image_path + data['file_name'][0], copy_path + data['file_name'][0].split('/')[-1])
    #
    # print(data['file_name'][0].split('/')[-1])