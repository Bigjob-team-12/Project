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

def main(img_path,saved_model_loaded):

    config = ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4
    #session = InteractiveSession(config=config)
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    #STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

    ###### option setting ######
    input_size = 416
    image_path = img_path
    print(image_path)
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    weights = './checkpoints/yolov4-416'
    iou = 0.45
    score = 0.25

    ###### data processing ######
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.
    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    ###### load model ######
    # model은 앞에 실행되는 스크립트에서 불러온다. (속도 향상을 위해)
    saved_model_loaded = saved_model_loaded
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

    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    #print(boxes.numpy(), scores.numpy(), classes.numpy())

    # utils.draw_bbox_and_crop 함수는 직접 custom한 함수
    image = utils.draw_bbox_and_crop(original_image, pred_bbox, size=224)
    #image = utils.draw_bbox(image_data*255, pred_bbox)   #original 함수
    image = Image.fromarray(image.astype(np.uint8))
    #image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite(image_path, image)

if __name__ == '__main__':
    try:
        app.run(main('./data/g_2.jpg'))
    except SystemExit:
        pass
