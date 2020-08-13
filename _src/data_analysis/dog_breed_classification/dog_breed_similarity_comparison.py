import pandas as pd
from PIL import Image
import os
from numpy import dot
from numpy.linalg import norm
import platform
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt

def init_font():
    '''
    한글 깨짐 방지
    :return: None
    '''
    if (platform.system() == 'Darwin'):
        rc('font', family='AppleGothic')
    elif (platform.system() == 'Windows'):
        path = 'c:/Windows/Fonts/malgun.ttf'
        font_name = font_manager.FontProperties(fname=path).get_name()
        rc('font', family=font_name)
    else:
        print('error')
def load_data():
    '''
    image별 softmax값 저장된 data 불러오기
    :return: data
    '''
    data = pd.read_csv('../../../_db/data/model_data/working/result.csv').set_index('file_name').iloc[:, 1:-3]

    print('-'*10 + 'data shape and data.head' + '-'*10)
    print(data.shape)
    print(data.head(2))

    return data
def load_similar_images(file_name, path):
    '''
    input image와 유사한 image 10개 추출
    :param file_name:
    :param path:
    :return: file list, image list
    '''
    # input image와 cosine 유사도가 높은 10개의 file list
    file_lst = data.apply(lambda x: cos_sim(x, data.loc[file_name]), axis=1).sort_values(ascending=False)[:11].index
    # file list의 image 불러오기
    img_lst = []
    for i in file_lst:
        img_lst.append(Image.open(os.path.join(path, i)))

    print('-'*10 + 'load similar image file list' + '-'*10)
    [print(_) for _ in file_lst]

    return file_lst, img_lst
def draw_plot(file_lst, img_lst):
    '''
    show 10 similar images
    :param file_lst:
    :param img_lst:
    :return: None
    '''
    rows = 2
    cols = 5
    axes = []
    fig = plt.figure(figsize=(30, 10))
    for a in range(rows * cols):
        axes.append(fig.add_subplot(rows, cols, a + 1))
        subplot_title = (file_lst[a].split('.')[0].split('/')[-1])
        axes[-1].set_title(subplot_title)
        plt.imshow(img_lst[a])
    fig.tight_layout()
    plt.show()
def cos_sim(A, B):
    '''
    A,B의 cosine similarity
    :param A:
    :param B:
    :return: A,B의 cosine similarity
    '''
    return dot(A, B) / (norm(A) * norm(B))

if __name__ == '__main__':
    # 한글 깨짐 방지
    init_font()
    # data load
    data = load_data()

    path = '../../../_db/data/model_data/input/dog_data/ours_dog/test'
    file_name = '14_GOLDEN_RETRIEVER/골든_리트리버_경기-용인-2020-00463.jpg'
    # input image와 cosine 유사도가 높은 10개의 file, image 추출
    file_lst, img_lst = load_similar_images(file_name, path)
    # 10개의 data 보여주기
    draw_plot(file_lst, img_lst)