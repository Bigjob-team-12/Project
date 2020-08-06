import pymysql
import urllib.request
import os


# DB connect
conn = pymysql.connect(host='localhost', user='root', password='1234',
                       db='project', charset='utf8')

def load_image():
    curs = conn.cursor()
    sql = 'SELECT distinct image, number, kind from protect_animals_url1'
    curs.execute(sql)
    images = curs.fetchall()
    return images
def download_image(images):
    path = '../../_db/data/'

    for url, number, class_name in images:
        print(number)

        dog_cat, *class_name = class_name.split()

        tmp_path = path + dog_cat
        # file 유무 확인 후 없을 경우 생성
        if not os.path.isdir(tmp_path):
            os.mkdir(tmp_path)

        class_name = '_'.join(class_name) if len(class_name) else 'none'

        tmp_path = tmp_path + '/' + class_name
        if not os.path.isdir(tmp_path):
            os.mkdir(tmp_path)
            
        urllib.request.urlretrieve(url, tmp_path + '/' + number + ".jpg")

if __name__ == '__main__':
    images = load_image()
    download_image(images)
