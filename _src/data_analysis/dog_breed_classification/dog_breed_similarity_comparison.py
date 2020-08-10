import pandas as pd
from PIL import Image
import os
from numpy import dot
from numpy.linalg import norm
import platform
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt

def init_font():
    if (platform.system() == 'Darwin'):
        rc('font', family='AppleGothic')
    elif (platform.system() == 'Windows'):
        path = 'c:/Windows/Fonts/malgun.ttf'
        font_name = font_manager.FontProperties(fname=path).get_name()
        rc('font', family=font_name)
    else:
        print('error')
def read_data():
    data = pd.read_csv('../../../_db/data/model_data/working/result.csv').set_index('file_name').iloc[:, 1:-3]

    print('-'*10 + 'data shape and data.head' + '-'*10)
    print(data.shape)
    print(data.head(2))

    return data
def load_similar_images(file_name, path):
    file_lst = data.apply(lambda x: cos_sim(x, data.loc[file_name]), axis=1) \
                  .sort_values(ascending=False)[:11].index
    img_lst = []
    for i in file_lst:
        img_lst.append(Image.open(os.path.join(path, i)))

    print('-'*10 + 'load similar image file list' + '-'*10)
    [print(_) for _ in file_lst]

    return file_lst, img_lst
def draw_plot(file_lst, img_lst):
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
    return dot(A, B) / (norm(A) * norm(B))

if __name__ == '__main__':
    init_font()

    data = read_data()

    path = '../../../_db/data/model_data/input/dog_data/ours_dog/test'
    file_name = '26_PEKINGESE/페키니즈_경기-수원-2020-00704.jpg'
    file_lst, img_lst = load_similar_images(file_name, path)

    draw_plot(file_lst, img_lst)