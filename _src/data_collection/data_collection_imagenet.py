import os
import urllib.request
from urllib.error import URLError, HTTPError
from tqdm import tqdm

import socket
socket.setdefaulttimeout(180)

# text 폴더에 저장된 text 파일에서 url list 추출
file_list = os.listdir('text')

for file in file_list:
    # text file명으로 폴더 생성
    directory = file.split('.')[0]
    os.makedirs(os.path.join(directory))

    i = 0
    with open('text/'+file, 'r', encoding='ISO-8859-1') as f:
        strings = f.readlines()
        print(directory, len(strings))

        # 각 url에서 이미지 저장
        for item in tqdm(strings):
            if len(item) > 3:
                try:
                    urllib.request.urlretrieve(item, directory+'/{}.jpg'.format(i))
                    i += 1
                except Exception as e:
                    #print(i, e)
                    pass