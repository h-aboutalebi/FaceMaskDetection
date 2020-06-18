import xml.etree.ElementTree as ET
import cv2
from tensorflow_infer import inference
from load_model.tensorflow_loader import load_tf_model, tf_inference
from utils.anchor_generator import generate_anchors
import numpy as np

class Evaluation():

    def __init__(self,thresh_iou,image_names):
        self.thresh_iou=thresh_iou
        self.image_names=image_names

    def evaluate(self,image_files,targets):
        sess, graph = load_tf_model('models/face_mask_detection.pb')
        # anchor configuration
        feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
        anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
        anchor_ratios = [[1, 0.62, 0.42]] * 5
        # generate anchors
        anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
        # for inference , the batch size is 1, the model output shape is [1, N, 4],
        # so we expand dim for anchors to [1, anchor_num, 4]
        anchors_exp = np.expand_dims(anchors, axis=0)
        id2class = {0: 'Mask', 1: 'NoMask'}

        for image in self.image_names:
            imgPath=image_files[image]
            img = cv2.imread(imgPath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            output_info=inference(img, id2class,anchors_exp,sess, graph,show_result=True, target_shape=(260, 260))



