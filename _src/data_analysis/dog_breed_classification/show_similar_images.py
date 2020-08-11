import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image
import cv2
import os
from dog_breed_similarity_comparison import read_data, cos_sim, draw_plot, init_font

def get_data_sets(dir, image_size):
    data = []
    files = []
    dir_lists = os.listdir(dir)

    for image_dir in dir_lists:
        file_path = os.path.join(dir, image_dir)
        if os.path.isfile(file_path):
            ff = np.fromfile(file_path, np.uint8)
            img = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED)

            image_array = Image.fromarray(img, 'RGB')

            resize_img = image_array.resize((image_size, image_size))

            data.append(np.array(resize_img))
            files.append(image_dir)
    test_data = np.array(data)
    test_files = np.array(files)

    return test_data, test_files
def get_steps(test_data):
    length=test_data.shape[0]
    batches=[int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=80]
    batches.sort(reverse=True)
    t_batch_size=batches[0]
    t_steps=length/t_batch_size
    return t_steps, t_batch_size
def make_model(rand_seed):
    size = 20
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

    return model
def make_generators(test_data, t_batch_size):
    test_datagen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,
                                    samplewise_center=True,
                                    samplewise_std_normalization=True)
    test_gen=test_datagen.flow(test_data, batch_size=t_batch_size, shuffle=False)

    return test_gen
def make_predictions(output_dir, test_gen, t_steps, model):
    # the best model was saved as a file need to read it in and load it since it is not available otherwise
    test_gen.reset()
    model_path=os.path.join(output_dir,'tmp.h5')
    model.load_weights(model_path)
    pred=model.predict_generator(test_gen, steps=t_steps,verbose=1) # make predictions on the test set
    return [pred, model]
def compare_similarities_and_show_results(predict, n=10):
    data = read_data()
    file_list = data.apply(lambda x: cos_sim(x, predict[0][0]), axis=1).sort_values(ascending=False).index[:100]
    image_path = '../../../_db/data/model_data/input/dog_data/ours_dog/test'
    img_lst = []
    for i in file_list:
        img_lst.append(Image.open(os.path.join(image_path, i)))
    init_font()
    draw_plot(file_list[:n], img_lst[:n])
def TF2_classify(source_dir,output_dir,image_size=224,rand_seed=128):
    test_data, test_files = get_data_sets(source_dir,image_size)
    t_steps, t_batch_size = get_steps(test_data)
    model=make_model(rand_seed)
    test_gen=make_generators(test_data, t_batch_size)
    predict=make_predictions(output_dir, test_gen, t_steps,model)
    compare_similarities_and_show_results(predict)

if __name__ == '__main__':
    source_dir='../../../_db/data/model_data/test'
    output_dir='../../../_db/data/model_data/working'
    image_size=224
    rand_seed=256

    TF2_classify(source_dir,output_dir,image_size=image_size,rand_seed=rand_seed)