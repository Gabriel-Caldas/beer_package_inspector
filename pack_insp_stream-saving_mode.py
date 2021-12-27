import numpy as np
import os
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# cpu use only, unable GPU, if available
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# cap = cv2.VideoCapture(2)
cap = cv2.VideoCapture('Video2.mp4')

# cam settings
# cap.set(3, 600)  # 3 = id for width
# cap.set(4, 400)  # 4 = id for height
cap.set(10, 100)  # 10 = id for bright
cap.set(5, 120)  # 5 = id for FPS

min_score = 70

# ## Object detection imports
# Here are the imports from the object detection module.
from utils import label_map_util
from utils import visualization_utils_sv as vis_util

# Model preparation
MODEL_NAME = 'inspection_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 3

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that
# this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate
# string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

## Tensorflow expects input which its formatted in NHWC format:
## (BATCH, HEIGHT, WIDTH, CHANNELS)

def load_image_into_numpy_array(image):
    last_axis = -1
    dim_to_repeat = 2
    repeats = 3
    grscale_img_3ds = np.expand_dims(image, last_axis)
    training_image = np.repeat(grscale_img_3ds, repeats, dim_to_repeat).astype('uint8')

    assert len(training_image.shape) == 3, "1"
    assert training_image.shape[-1] == 3, "2"
    return training_image




# # Detection

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
class_index = 0
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:

            ret, image_np = cap.read()
            # image_np = cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)
            # image_np = load_image_into_numpy_array(image_np)

            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # each box represents a part of the image where a particular object was detected
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # each score represents the level of confidence fpr each of the objects
            # score is show on the result label, together with the class label
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # noinspection PyRedeclaration
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual Detection
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded}
            )

            # Visualization of the results of a detection.
            _, class_index = vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                class_index,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                min_score_thresh=min_score/100,
                use_normalized_coordinates=True,
                line_thickness=2,
            )

            cv2.imshow('Pack Inspection', image_np)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
