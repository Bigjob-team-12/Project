import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout,Conv2D, MaxPooling2D,BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model, Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from PIL import Image
import cv2
import os
from tqdm import tqdm
import random as rd

def get_data_sets(dir, output_dir, subject, image_size, t_split, v_split, rand_seed):
    data = []
    labels = []
    class_list = []
    files = []
    net_split = (t_split + v_split) / 100
    v_share = t_split / (t_split + v_split)
    t_split = t_split / 100
    v_split = v_split / 100
    d_path = os.path.join(dir, "train")
    source_list = os.listdir(d_path)
    for s in source_list:
        s_path = os.path.join(d_path, s)
        if os.path.isdir(s_path):  # only process  directories not files
            class_list.append(s)
    class_count = len(class_list)  # determine number of class directories in order to set leave value intqdm
    train_path = os.path.join(dir, 'train')
    tr_data = read_files(train_path, class_list, image_size)
    # tr_data[0]=training files, tr_data[1]=training labels
    for i in range(0, 3):
        rd.seed(rand_seed)
        rd.shuffle(
            tr_data[i])  # shuffle training data but keep the labels tied to the array by using same seed
    test_path = os.path.join(dir, 'test')
    t_data = read_files(test_path, class_list, image_size)
    # t_data[0] test files, t_data[1]=test labels
    valid_path = os.path.join(dir, 'valid')
    v_data = read_files(valid_path, class_list, image_size)
    # shuffle the validation images but keep labels tied to the file by using same random seed
    for i in range(0, 3):
        rd.seed(rand_seed)
        rd.shuffle(v_data[i])
    train_data = np.array(tr_data[0])
    train_labels = np.array(tr_data[1])
    train_files = np.array(tr_data[2])
    val_data = np.array(v_data[0])
    val_labels = np.array(v_data[1])
    val_files = np.array(v_data[2])
    test_data = np.array(t_data[0])
    test_labels = np.array(t_data[1])
    test_files = np.array(t_data[2])
    print_data(train_labels, test_labels, val_labels, class_list)
    # save the class dictionary as a text file so it can be used by classification.py in the future
    msg = ''
    for i in range(0, class_count):
        msg = msg + str(i) + ':' + class_list[i] + ','
    id = subject + '.txt'
    dict_path = os.path.join(output_dir, id)
    print('\n saving dictionary of class names and labels to {0}'.format(dict_path))
    f = open(dict_path, 'w')
    f.write(msg)
    f.close()

    return [train_data, train_labels, test_data, test_labels, val_data, val_labels, test_files, class_list]
def read_files(dir_path, class_list, image_size):
    data = []
    labels = []
    files = []
    if (len(class_list) <= 6):
        leave = True
    else:
        leave = False
    dir_list = os.listdir(dir_path)  # get list of items in dir
    class_count = -1
    for d in dir_list:  # these should be the class directories
        d_path = os.path.join(dir_path, d)  # path to class directories
        if os.path.isdir(d_path):  # only process directories ignore any files in dir
            d_list = d_path.split(r'/')
            length = len(d_list)
            desc = d_list[length - 2] + '-' + d_list[length - 1]
            class_count = class_count + 1
            file_list = os.listdir(d_path)  # list contents of directory d , they should just be files but better to check
            for f in tqdm(file_list, desc=desc, unit='files', leave=leave):
                file_path = os.path.join(d_path, f)
                if os.path.isfile(file_path):  # just process files
                    index = f.rfind('.')
                    ext = f[index + 1:].lower()  # get the file's extension
                    if ext in ['jpg', 'jpeg', 'jpe', 'jp2', 'tiff', 'png']:  # make sure image formats work with cv2
                        # 한글 경로 인식 문제
                        ff = np.fromfile(file_path, np.uint8)
                        img = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED)

                        image_array = Image.fromarray(img, 'RGB')

                        resize_img = image_array.resize((image_size, image_size))
                        data.append(np.array(resize_img))  # data is a list of image arrays for the entire data set
                        labels.append(class_count)  # class count is the label associated with the array image
                        files.append(f)

    return [data, labels, files]
def print_data(train_labels, test_labels, val_labels, class_list):
    train_list=list(train_labels)
    test_list=list(test_labels)
    val_list=list(val_labels)
    print('{0:9s}Class Name{0:10s}Class No.{0:4s}Train Files{0:7s}Test Files{0:5s}Valid Files'.format(' '))
    for i in range(0, len(class_list)):
        c_name=class_list[i]
        tr_count=train_list.count(i)
        tf_count=test_list.count(i)
        v_count=val_list.count(i)
        print('{0}{1:^25s}{0:5s}{2:3.0f}{0:9s}{3:4.0f}{0:15s}{4:^4.0f}{0:12s}{5:^3.0f}'.format(' ',
                                                                                               c_name,i,tr_count,
                                                                                               tf_count,v_count))
    print('{0:30s} ______________________________________________________'.format(' '))
    msg='{0:10s}{1:6s}{0:16s}{2:^3.0f}{0:8s}{3:3.0f}{0:15s}{4:3.0f}{0:13s}{5}\n'
    print(msg.format(' ', 'Totals',len(class_list),len(train_labels),len(test_labels),len(val_labels)))
def get_steps(test_data):
    length=test_data.shape[0]
    batches=[int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=80]
    batches.sort(reverse=True)
    t_batch_size=batches[0]
    t_steps=length/t_batch_size
    return t_steps, t_batch_size
def make_model(class_list, rand_seed, lr_factor):
    size = len(class_list)
    check_file = os.path.join(output_dir, 'tmp.h5')
    mobile = tf.keras.applications.mobilenet.MobileNet()
    # remove last 5 layers of model and add dense layer with 128 nodes and the prediction layer with size nodes
    # where size=number of classes
    x = mobile.layers[-6].output
    x = Dense(128, kernel_regularizer=regularizers.l2(l=0.015), activation='relu')(x)
    x = Dropout(rate=.5, seed=rand_seed)(x)
    predictions = Dense(size, activation='softmax')(x)
    model = Model(inputs=mobile.input, outputs=predictions)
    for layer in model.layers:
        layer.trainable = True
    model.compile(Adam(lr=0.0015), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, mode='min', verbose=1)
    checkpoint = ModelCheckpoint(check_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
    lrck = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=1,
                                             verbose=1, mode='min', min_delta=0.000001, cooldown=1, min_lr=1.0e-08)
    callbacks = [checkpoint, lrck, early_stop]
    return [model, callbacks]
def make_generators(data_sets, t_batch_size):
    test_datagen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,
                             samplewise_center=True,
                             samplewise_std_normalization=True)
    test_gen=test_datagen.flow(data_sets[2], data_sets[3], batch_size=t_batch_size, shuffle=False)

    return test_gen
def make_predictions(output_dir, test_gen, t_steps, model_data):
    # the best model was saved as a file need to read it in and load it since it is not available otherwise
    test_gen.reset()
    msg='Training has completed, now loading saved best model and processing test set to see how accurate the model is'
    print (msg,flush=True)
    model_path=os.path.join(output_dir,'tmp.h5')
    model = model_data[0]
    model.load_weights(model_path)
#     model=load_model(model_path) # load the saved model with lowest validation loss / ,custom_objects={'KerasLayer':hub.KerasLayer}
    pred=model.predict_generator(test_gen, steps=t_steps,verbose=1) # make predictions on the test set
    return [pred, model]
def display_pred(output_dir, pred, t_files, t_labels, class_list, subject):
    # t_files are the test files, t_labels are the class label associated with the test file
    # class_list is a list of classes
    trials = len(t_files)  # number of predictions made should be same as len(t_files)
    errors = 0
    prob_list = []
    true_class = []
    pred_class = []
    file_list = []
    x_list = []
    index_list = []
    pr_list = []
    error_msg = ''
    for i in range(0, trials):
        p_c_num = pred[i].argmax()  # the index with the highest prediction value
        if p_c_num != t_labels[i]:  # if the predicted class is not the same as the test label it is an error
            errors = errors + 1
            file_list.append(t_files[i])  # list of file names that are in error
            true_class.append(class_list[t_labels[i]])  # list classes that have an eror
            pred_class.append(class_list[p_c_num])  # class the prediction selected
            prob_list.append(100 * pred[i][p_c_num])  # probability of the predicted class
            add_msg = '{0:^24s}{1:5s}{2:^20s}\n'.format(class_list[t_labels[i]], ' ', t_files[i])
            error_msg = error_msg + add_msg

    accuracy = 100 * (trials - errors) / trials
    print('\n There were {0} errors in {1} trials for an accuracy of {2:7.3f}'.format(errors, trials, accuracy, ),
          flush=True)
    if errors <= 25:
        msg = '{0}{1:^24s}{0:3s}{2:^20s}{0:3s}{3:20s}{0:3s}{4}'
        print(msg.format(' ', 'File Name', 'True Class', 'Predicted Class', 'Probability'))
        for i in range(0, errors):
            msg = '{0}{1:^24s}{0:3s}{2:^20s}{0:3s}{3:20s}{0:5s}{4:^6.2f}'
            print(msg.format(' ', file_list[i], true_class[i], pred_class[i], prob_list[i]))
    else:
        print('with {0} errors the full error list will not be printed'.format(errors))
    acc = '{0:6.2f}'.format(accuracy)
    header = 'Classification subject: {0} There were {1} errors in {2} tests for an accuracy of {3} using a model\n'.format(
        subject, errors, trials, acc)
    header = header + '{0:^24s}{1:5s}{2:^20s}\n'.format('CLASS', ' ', 'FILENAME')
    error_msg = header + error_msg
    file_name = 'error list-' + acc + '.txt'
    print('\n file {0} containing the list of errors has been saved to {1}'.format(file_name, output_dir))
    file_path = os.path.join(output_dir, file_name)
    f = open(file_path, 'w')
    f.write(error_msg)
    f.close()
    for c in class_list:
        count = true_class.count(c)
        x_list.append(count)
        pr_list.append(c)
    for i in range(0, len(x_list)):  # only plot classes that have errors
        if x_list[i] == 0:
            index_list.append(i)
    for i in sorted(index_list, reverse=True):  # delete classes with no errors
        del x_list[i]
        del pr_list[i]  # use pr_list - can't change class_list must keep it fixed
    fig = plt.figure()
    fig.set_figheight(len(pr_list) / 4)
    fig.set_figwidth(6)
    plt.style.use('fivethirtyeight')
    for i in range(0, len(pr_list)):
        c = pr_list[i]
        x = x_list[i]
        plt.barh(c, x, )
        plt.title('Errors by class')
    plt.show()
    time.sleep(5.0)

    return accuracy
def save_predicted_value_as_csv(predict, data_sets):
    class_list = data_sets

    result = pd.DataFrame(predict[0], columns=class_list[7])

    result['true_class'] = class_list[3]
    result['predict_class'] = result.iloc[:, :-1].idxmax(axis=1)
    result['class_name'] = list(map(lambda x: class_list[7][x], result['true_class']))
    result['file_name'] = result['class_name'] + '/' + class_list[6]

    result.head()
    result.to_csv(output_dir + '/result.csv')
def TF2_classify(source_dir,output_dir,subject, t_split=10, v_split=5, batch_size=80,lr_factor=.8,image_size=224,rand_seed=128):

    data_sets=get_data_sets(source_dir,output_dir,subject,image_size,t_split, v_split, rand_seed)
    t_steps, t_batch_size=get_steps(data_sets[2])
    model_data=make_model(data_sets[7], rand_seed,lr_factor)
    test_gen=make_generators(data_sets,t_batch_size)
    predict=make_predictions(output_dir, test_gen, t_steps,model_data)
    accuracy=display_pred(output_dir,predict[0],data_sets[6],data_sets[3],data_sets[7], subject)
    print('-'*10 + 'accuracy : ' + str(accuracy) + ' ' + '-'*10)
    save_predicted_value_as_csv(predict, data_sets)

if __name__ == '__main__':
    source_dir='../../../_db/data/model_data/input/dog_data/ours_dog'
    output_dir='../../../_db/data/model_data/working'
    subject='dog_breeds'
    t_split=8
    v_split=8
    batch_size=96
    image_size=224
    rand_seed=256

    TF2_classify(source_dir,output_dir,subject, t_split=t_split, v_split=v_split, batch_size=batch_size,image_size=image_size,rand_seed=rand_seed)