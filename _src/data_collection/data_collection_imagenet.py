import os
import urllib.request
from urllib.error import URLError, HTTPError
from tqdm import tqdm

import socket
socket.setdefaulttimeout(180)

file_list = os.listdir('text')
for file in file_list:
    directory = file.split('.')[0]
    os.makedirs(os.path.join(directory))
    i = 0
    with open('text/'+file, 'r', encoding='ISO-8859-1') as f:
        strings = f.readlines()
        print(directory, len(strings))
        for item in tqdm(strings):
            if len(item) > 3:
                try:
                    urllib.request.urlretrieve(item, directory+'/{}.jpg'.format(i))
                    i += 1
                except Exception as e:
                    #print(i, e)
                    pass