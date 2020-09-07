#-*- coding:utf-8 -*-
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
import time
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('image', './data/g_0.jpg', 'path to input image')
flags.DEFINE_string('output', 'result.png', 'path to output image')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')



# 한국어 지원 이미지 읽기 cv2
def hangulFilePathImageRead(filePath) :

    stream = open( filePath.encode("utf-8") , "rb")
    bytes = bytearray(stream.read())
    numpyArray = np.asarray(bytes, dtype=np.uint8)

    return cv2.imdecode(numpyArray , cv2.IMREAD_UNCHANGED)

# 한국어 경로 지원되는 image write 함수
def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)
        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False

def main(img_path,saved_model_loaded):
    # path_file = FLAGS.path_file
    #
    # for count, foldername in enumerate(os.listdir(path_file)):
    #     print(count, foldername)
    #     for count, filename in enumerate(os.listdir(path_file + '/' + foldername)):
    #         print(count, filename)
    #         dst_name = filename.split('/')[-1].split('.jpg')[0] + '_crop' + '.jpg'
    #         path_src_file = path_file + '/' + foldername + '/' + filename
    #         path_dst_file = path_file + '/' + foldername + '/' + dst_name
    #         print(path_src_file)

    config = ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4
    #session = InteractiveSession(config=config)
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


    #STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

    input_size = 416
    image_path = img_path
    print(image_path)
    # 한글 경로 인식 문제
    # ff = np.fromfile(image_path, np.uint8)
    # original_image = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED)
    # original_image = Image.fromarray(original_image, 'RGB')
    original_image = hangulFilePathImageRead(image_path)
    #original_image = cv2.imread(image_path)

    try:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(str(e))
        error = 1
        return error
    weights = './checkpoints/yolov4-416'
    iou = 0.45
    score = 0.25

    # image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.
    # image_data = image_data[np.newaxis, ...].astype(np.float32)

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    # if FLAGS.framework == 'tflite':
    #     interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
    #     interpreter.allocate_tensors()
    #     input_details = interpreter.get_input_details()
    #     output_details = interpreter.get_output_details()
    #     print(input_details)
    #     print(output_details)
    #     interpreter.set_tensor(input_details[0]['index'], images_data)
    #     interpreter.invoke()
    #     pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    #     if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
    #         boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
    #     else:
    #         boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
    # else:
    #start = time.time()
    #saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
    saved_model_loaded = saved_model_loaded
    #print('model load time = ', time.time() - start)


    infer = saved_model_loaded.signatures['serving_default']
    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold= iou,
        score_threshold= score
    )
    # print(boxes.numpy()[0][0])
    #
    #print(classes.numpy()[0][0])
    # error=0
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    #rint(pred_bbox)
    image = utils.draw_bbox_and_crop(original_image, pred_bbox, size=224)
    #image = utils.draw_bbox(image_data*255, pred_bbox)
    # if error == 1:
    #     print('dog 검출 x')

    image = Image.fromarray(image.astype(np.uint8))
    #image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    #cv2.imwrite(image_path, image)
    imwrite(image_path, image)

if __name__ == '__main__':
    try:
        app.run(main('./data/g_2.jpg'))
    except SystemExit:
        pass
