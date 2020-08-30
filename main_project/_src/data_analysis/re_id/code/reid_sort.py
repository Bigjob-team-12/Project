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
import matplotlib.pyplot as plt


#####################################################################
# Show result
# ----------------------
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
# ----------------------
def sort_img(qf, gf):
    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()

    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    num = len(index)
    return index, score, num


#######################################################################
# main code
# ----------------------
def main(result, image_datasets, mode, first=True):
    ######################################################################
    # Options
    # --------
    query_index = 0

    if mode == 'all':  # 전체 공고대상 검색
        print('all')
        test_dir = 'C:/Users/kdan/BigJob12/main_project/_db/data'  # 전체
        mode_path = 'preprocessed_data'
        # gallery_result = scipy.io.loadmat('C:/Users/kdan/BigJob12/main_project/_src/data_analysis/re_id/code/gallery_all_result.mat')

        gallery_all_result = pd.read_csv(
            'C:/Users/kdan/BigJob12/main_project/_src/data_analysis/re_id/code/gallery_all_result.csv').iloc[:, 1:]
        gallery_result = {'gallery_f': np.array(gallery_all_result, dtype=np.float32)}

    else:
        print('not_all')
        test_dir = 'C:/Users/kdan/BigJob12/main_project/_db/data/model_data'
        mode_path = 'gallery'
        reid_gallery.main('not_all')
        gallery_result = scipy.io.loadmat(
            'C:/Users/kdan/BigJob12/main_project/_src/data_analysis/re_id/code/gallery_result.mat')
        # reid_gallery.main('not_all')

    # load data
    data_dir = test_dir
    image_datasets = image_datasets
    result = result

    image_datasets_gallery = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in [mode_path]}
    query_feature = torch.FloatTensor(result['query_f'])
    gallery_feature = torch.FloatTensor(gallery_result['gallery_f'])
    query_feature = query_feature
    gallery_feature = gallery_feature
    # query_feature = query_feature.cuda()  # use gpu
    # gallery_feature = gallery_feature.cuda() # use gpu

    i = query_index
    index, score, num = sort_img(query_feature[i], gallery_feature)
    print('sort index= ', index)

    ########################################################################
    # Visualize the rank result
    query_path, _ = image_datasets['query'].imgs[i]
    print(query_path)
    print('Top N images are as follow:')
    try:
        if num > 10:
            re_id_result = [image_datasets_gallery[mode_path].imgs[index[_]][0].split('\\')[-1][:-4].split('_')[-1] for
                            _ in range(1000)]
            tmp_result = pd.read_csv('C:/Users/kdan/BigJob12/main_project/_db/data/model_data/working/to_reid.csv')[
                'file_name'].apply(lambda x: x.split('_')[-1][:-4]).tolist()
            result = []
            for _ in re_id_result:
                if _ in tmp_result: result.append(_)
            if first:  # email
                pd.DataFrame(result).to_csv(
                    'C:/Users/kdan/BigJob12/main_project/_db/data/model_data/working/to_email.csv')
            pd.DataFrame(result).to_csv('C:/Users/kdan/BigJob12/main_project/_db/data/model_data/working/to_web.csv')

        else:
            result = [image_datasets_gallery[mode_path].imgs[index[_]][0].split('\\')[-1][:-4].split('_')[-1] for _ in
                      range(num)]
            pd.DataFrame(result).to_csv('C:/Users/kdan/BigJob12/main_project/_db/data/model_data/working/to_web.csv')

        # show Demo image
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

    # fig.savefig("show.png")


if __name__ == '__main__':
    main()