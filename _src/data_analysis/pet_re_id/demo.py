import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib

matplotlib.use('agg')
import cv2
import matplotlib.pyplot as plt

#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_index', default=0, type=int, help='test_image_index')
parser.add_argument('--test_dir', default='./pytorch_re_id', type=str, help='./test_data')
opts = parser.parse_args()

data_dir = opts.test_dir
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in ['gallery', 'query']}


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


######################################################################
result = scipy.io.loadmat('pytorch_result.mat')
query_feature = torch.FloatTensor(result['query_f'])
# query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
# gallery_label = result['gallery_label'][0]

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()


#######################################################################
# sort the images
# def sort_img(qf, ql, gf, gl ):

def sort_img(qf, gf):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    print(score)
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    print(index)
    return index, score


i = opts.query_index
index, score = sort_img(query_feature[i], gallery_feature)
# index, score = sort_img(query_feature[i],query_label[i],gallery_feature,gallery_label)
print('sort index= ', index)
########################################################################
# Visualize the rank result

query_path, _ = image_datasets['query'].imgs[i]
# query_label = query_label[i]
print(query_path)
print('Top 20 images are as follow:')
try:  # Visualize Ranking Result
    # Graphical User Interface is needed
    fig = plt.figure(figsize=(16, 8))
    ax = plt.subplot(2, 11, 1)
    ax.axis('off')
    imshow(query_path, 'query')
    for i in range(20):
        ax = plt.subplot(2, 11, i + 2)
        ax.axis('off')
        img_path, _ = image_datasets['gallery'].imgs[index[i]]
        #   label = gallery_label[index[i]]

        imshow(img_path)
        ax.set_title('%d : %0.4f' % (i + 1, score[index[i]]), color='red')
        # cv2.imwrite('./%d.jpg' %(i), img_path)
        #   if label == query_label:
        #       ax.set_title('%d : %0.4f'%(i+1, score[index[i]]), color='green')
        #  else:
        #       ax.set_title('%d : %0.4f'%(i+1, score[index[i]]), color='red')

        print(img_path, index[i], score[index[i]])
except RuntimeError:
    for i in range(20):
        img_path = image_datasets.imgs[index[i]]
        print(img_path[0])
    print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

fig.savefig("show.png")
