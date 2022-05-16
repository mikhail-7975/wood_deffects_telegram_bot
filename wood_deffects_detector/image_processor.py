import cv2
import numpy as np
from wood_deffects_detector.onnx_detection_model import YoloOnnxDetector

IMG_SIZE = 1024


class DefectsDetector:
    def __init__(self, detection_model_path):
        self.img_size = 256
        self.model = YoloOnnxDetector(detection_model_path)

    def preprocess(self, img):
        img = cv2.resize(img, (self.img_size, self.img_size))
        return img

    def run(self, img):
        inp_img = self.preprocess(img)
        bboxes = self.model.run(inp_img)
        result_img = self.highlight_deffects(img, bboxes)
        return result_img

    def highlight_deffects(self, img, deffects_bboxes):
        img_ = img.copy()
        for bbox in deffects_bboxes:
            *xyxy, conf, cls = bbox
            x1, y1, x2, y2 = list(map(int, xyxy))
            cv2.rectangle(img_, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
        return img_
