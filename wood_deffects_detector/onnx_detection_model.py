import typing as t
from pathlib import Path

import cv2
import numpy as np
import torch
from onnxruntime import InferenceSession

from wood_deffects_detector.utils import letterbox
from wood_deffects_detector.utils import non_max_suppression, scale_coords


class YoloOnnxDetector:
    """
    Person detection class
    return: list with bboxes and confidence for each person on photo
     """
    def __init__(self, onnx_model_path, classes=None):
        self.conf_thres = 0.3
        self._pd_threshold = 0.3
        self.model = InferenceSession(onnx_model_path)
        self.classes = classes

    def preprocess(self, img: np.ndarray):
        img = (img / 255.).astype(np.float32)
        img = cv2.resize(img, (640, 640))
        img = img.transpose(2, 0, 1)
        img = img[None]
        return img

    def run(self, img0):
        """
        :param img0: input image
        :return: list( torch.tensor(shape=(n, 5)) ) xyxy, conf
        """
        session = self.model
        img_output = []
        img = letterbox(img0, stride=32, auto=False)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = img.astype('float32')
        img /= 255.0
        if len(img.shape) == 3:
            img = img[None]
        pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
        pred = non_max_suppression(pred, self.conf_thres, self._pd_threshold, self.classes, False, max_det=1000)
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    img_output.append(xyxy + [conf, cls])
        return img_output
