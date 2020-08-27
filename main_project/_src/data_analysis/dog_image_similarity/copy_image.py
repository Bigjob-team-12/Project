import shutil
import pandas as pd
import time
import sys
import gc
import torch
from numba import cuda
import os

def main():
    data = pd.read_csv('C:/Users/kdan/BigJob12/main_project/_db/data/model_data/working/to_reid.csv')

    image_path = 'C:/Users/kdan/BigJob12/main_project/_db/data/Preprocessed_data/'

    copy_path = 'C:/Users/kdan/BigJob12/main_project/_db/data/model_data/gallery/gallery_list/'

    shutil.rmtree(copy_path)

    if not os.path.isdir(copy_path[:-1]):
        os.mkdir(copy_path[:-1])

    start = time.time()  # 시작 시간 저장

    for image_file_path in data['file_name']:
        try:
            shutil.copy(image_path + image_file_path, copy_path + image_file_path.split('/')[-1])
        except:
            pass

    print(len(data['file_name']))
    print("time :", time.time() - start)

    gc.collect()
    sys.stdout.flush()
    cuda.close()
    torch.cuda.empty_cache() # PyTorch thing

if __name__ == '__main__':
    main()
