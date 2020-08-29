import argparse
import scipy.io
import torch
import numpy as np
import os
import pandas as pd
from torchvision import datasets
import matplotlib
import reid_gallery
matplotlib.use('agg')
import cv2
import matplotlib.pyplot as plt
from numba import cuda
#####################################################################
# Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    plt.savefig("demo.png")
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

#######################################################################
# sort the images
# def sort_img(qf, ql, gf, gl ):

def sort_img(qf, gf):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    #print(score)

    # predict index

    index = np.argsort(score)  # from small to large
    index = index[::-1]
    #print(index)
    num = len(index)
    return index, score, num


def main(result,image_datasets,mode, first = True):
    ######################################################################
    # Options
    # --------
    query_index = 0
    #test_dir= 'C:/Users/kdan/BigJob12/main_project/_db/data/model_data'
    #test_dir='C:/Users/kdan/BigJob12/main_project/_db/data'  # 전체

    if mode == 'all':
        print('all')
        test_dir = 'C:/Users/kdan/BigJob12/main_project/_db/data'  # 전체
        mode_path = 'preprocessed_data'
        # gallery_result = scipy.io.loadmat('C:/Users/kdan/BigJob12/main_project/_src/data_analysis/re_id/code/gallery_all_result.mat')

        test = pd.read_csv('C:/Users/kdan/BigJob12/main_project/_src/data_analysis/re_id/code/gallery_all_result.csv').iloc[:, 1:]
        gallery_result = {'gallery_f': np.array(test, dtype=np.float32)}

        # print(gallery_result)
        # print(type(gallery_result))


    else:
        print('not_all')
        test_dir = 'C:/Users/kdan/BigJob12/main_project/_db/data/model_data'
        mode_path = 'gallery'
        reid_gallery.main('not_all')
        gallery_result = scipy.io.loadmat('C:/Users/kdan/BigJob12/main_project/_src/data_analysis/re_id/code/gallery_result.mat')
        #reid_gallery.main('not_all')

        # load data
    data_dir = test_dir
    image_datasets = image_datasets
    result = result


    image_datasets_gallery = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in [mode_path]}
    print(mode_path)
    print(image_datasets_gallery)
    query_feature = torch.FloatTensor(result['query_f'])
    gallery_feature = torch.FloatTensor(gallery_result['gallery_f'])

    # query_feature = query_feature.cuda()
    # gallery_feature = gallery_feature.cuda()
    query_feature = query_feature
    gallery_feature = gallery_feature



    i = query_index
    index, score, num = sort_img(query_feature[i], gallery_feature)
    print('sort index= ', index)
    ########################################################################
    # Visualize the rank result
    query_path, _ = image_datasets['query'].imgs[i]
    print(query_path)

    print('Top 10 images are as follow:')
    try:
        if num > 10:
            re_id_result = [image_datasets_gallery[mode_path].imgs[index[_]][0].split('\\')[-1][:-4].split('_')[-1]  for _ in range(500)]
            print(re_id_result)

            
            #### 수정 필요
            tmp_result = pd.read_csv('C:/Users/kdan/BigJob12/main_project/_db/data/model_data/working/to_reid.csv')['file_name'].apply(lambda x : x.split('_')[-1][:-4]).tolist()

            print(tmp_result)

            result = []

            for _ in re_id_result:
                if _ in tmp_result: result.append(_)

            print(result)

            if first:
                pd.DataFrame(result).to_csv('C:/Users/kdan/BigJob12/main_project/_db/data/model_data/working/to_email.csv')
            pd.DataFrame(result).to_csv('C:/Users/kdan/BigJob12/main_project/_db/data/model_data/working/to_web.csv')

        else:
            result = [image_datasets_gallery[mode_path].imgs[index[_]][0].split('\\')[-1][:-4].split('_')[-1]  for _ in range(num)]
            print(result)
            pd.DataFrame(result).to_csv('C:/Users/kdan/BigJob12/main_project/_db/data/model_data/working/to_web.csv')

        #####DEBUG 용#####
        # for i in range(10):
        #     ax = plt.subplot(1, 11, i + 2)
        #     ax.axis('off')
        #     img_path, _ = image_datasets['gallery'].imgs[index[i]]
        #     ax.set_title('%d : %0.4f' % (i+1, score[index[i]]), color='red')
        #     # cv2.imwrite('./%d.jpg' %(i), img_path)
        #     print(img_path, index[i], score[index[i]])

    except RuntimeError:
        for i in range(20):
            img_path = image_datasets.imgs[index[i]]
            print(img_path[0])
        print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

    #fig.savefig("show.png")



if __name__ == '__main__':
    main()