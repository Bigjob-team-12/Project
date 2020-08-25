import shutil
import os

if __name__ == '__main__':
    start_path = '../../../_db/data/Crawling_data/[개]/'
    end_path = '../../../_db/data/Preprocessed_data/[개]/'

    class_lst = [_ for _ in os.listdir(start_path)]

    for _class in class_lst:
        if not os.path.isdir(end_path + _class):
            os.mkdir(end_path + _class)

        file_lst = os.listdir(start_path + _class)

        for file in file_lst:
            print(file)
            shutil.move(start_path + _class + '/' + file,end_path + _class + '/' + file)