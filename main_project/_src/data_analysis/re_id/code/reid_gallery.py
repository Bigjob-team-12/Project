# -*- coding: utf-8 -*-

from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
import os
import scipy.io
import yaml
import math
from model import ft_net
from PIL import ImageFile
import pandas as pd
import numpy as np
import demo
from numba import cuda
import time
######################################################################
# Load model
# ---------------------------
def load_network(network,name,which_epoch):
    save_path = os.path.join('C:/Users/kdan/BigJob12/main_project/_src/data_analysis/re_id/code/model', name, 'net_%s.pth' % which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network

######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feature(model, dataloaders,ms):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        # count += n
        # print(count)
        ff = torch.FloatTensor(n, 512).zero_().cuda()

        #     start = time.time()
        for i in range(2):
            if (i == 1):
                img = fliplr(img)

            input_img = Variable(img.cuda())

            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic',
                                                          align_corners=False)
                outputs = model(input_img)
                ff += outputs
        #    print("time = ",(time.time() - start)*1000)
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff.data.cpu()), 0)
    return features


def main(mode):
    cuda.select_device(0)
    cuda.close()

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    ######################################################################
    # Options
    # --------
    gpu_ids='0'
    which_epoch='last'
    if mode=='all':
        test_dir = 'C:/Users/kdan/BigJob12/main_project/_db/data'  # 전체
        mode_path = 'preprocessed_data'
    else:
        test_dir = 'C:/Users/kdan/BigJob12/main_project/_db/data/model_data'
        mode_path = 'gallery'

    name='ft_ResNet50'
    batchsize =32
    ms ='1'

    ###load config###
    # load the training config
    config_path = os.path.join('C:/Users/kdan/BigJob12/main_project/_src/data_analysis/re_id/code/model', name, 'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream)
    stride = config['stride']

    if 'nclasses' in config:  # tp compatible with old config files
        nclasses = config['nclasses']

    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)

    print('We use the scale: %s' % ms)
    str_ms = ms.split(',')
    ms = []
    for s in str_ms:
        s_f = float(s)
        ms.append(math.sqrt(s_f))

    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

    ######################################################################
    # Data transform
    data_transforms = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

    data_dir = test_dir

    ######################################################################
    # Load data
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in [mode_path]}

    # print(image_datasets)
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,
                                                  shuffle=False, num_workers=0) for x in [mode_path]}
    use_gpu = torch.cuda.is_available()

    ######################################################################
    # Load Collected data Trained model
    print('-------test-----------')
    model_structure = ft_net(nclasses, stride=stride)
    model = load_network(model_structure,name,which_epoch)


    # Remove the final fc layer and classifier layer
    model.classifier.classifier = nn.Sequential()
    print('model')

    # Change to test mode
    model = model.eval()
    if use_gpu:
        model = model.cuda()
    start_load = time.time()

    # Extract feature
    with torch.no_grad():
        gallery_feature = extract_feature(model, dataloaders[mode_path], ms)
    print(time.time() - start_load)

    # Save to Matlab for check
    gallery_result = {'gallery_f': gallery_feature.numpy()}

    print(type(gallery_result))
    print(gallery_result)

    if mode == 'all':
        scipy.io.savemat('C:/Users/kdan/BigJob12/main_project/_src/data_analysis/re_id/code/gallery_all_result.mat', gallery_result)
        pd.DataFrame(gallery_result['gallery_f']).to_csv(('gallery_all_result.csv'))
    else:
        scipy.io.savemat('C:/Users/kdan/BigJob12/main_project/_src/data_analysis/re_id/code/gallery_result.mat', gallery_result)
    print(name)

    #result = './model/%s/result.txt' % name

    # pd.DataFrame(gallery_result['gallery_f']).to_csv(('gallery_all_result.csv'))


    # gallery_result = scipy.io.loadmat('C:/Users/kdan/BigJob12/main_project/_src/data_analysis/re_id/code/gallery_all_result.mat')
    #
    # gallery_result2 = pd.read_csv('gallery_all_result.csv').iloc[:, 1:]
    # test = {'gallery_f' : np.array(gallery_result2, dtype=np.float32)}
    #
    # print(test)
    #
    # print(test == gallery_result)
    #
    # print(gallery_result['gallery_f'].shape)
    # print(test['gallery_f'].shape)
    #




if __name__ == '__main__':
    start = time.time()
    main('all')
    print(time.time() - start)
