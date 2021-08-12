import os

import numpy as np
import torch
from PIL import Image
from model_service.pytorch_model_service import PTServingBaseService as ParentClass

from modelarts.utils.augmentations import letterbox
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, \
    scale_coords
from utils.torch_utils import select_device


class InferenceService(ParentClass):
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
        self.class_names = {0: 'speed_unlimited', 1: 'red_stop', 2: 'speed_limited',
                            3: 'yellow_back', 4: 'green_go', 5: 'pedestrian_crossing'}
        self.device = select_device('cpu')
        self.model = attempt_load(self.model_path, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        self.half = False
        if self.half:
            self.model.half()  # to FP16

        self.infer_img_size = 640
        self.infer_img_size = check_img_size(self.infer_img_size, s=stride)  # check image size

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content)
                preprocessed_data[k] = img
        return preprocessed_data

    # 图片检测的推理过程：
    def _inference(self, data):
        """
        TODO 请在本函数中实现模型的推理过程，不同模型可根据不同的需要进行自定义修改
        """
        # 读取方式应该是Image进行读入的  也就是说 和我原先的代码读入方式 是完全相同的
        img0 = data['images']  # 本行代码必须保留，且无需修改
        # Padded resize
        img = letterbox(img0, self.infer_img_size, stride=self.stride)[0]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        with torch.no_grad():
            # do predict
            pred = self.model(img, augment=False, visualize=False)[0]
            # NMS
            pred = non_max_suppression(pred,
                                       conf_thres=0.35,
                                       iou_thres=0.45,
                                       classes=None,
                                       agnostic=False,
                                       max_det=100)

            # Process predictions
            for i, det in enumerate(pred):  # detections per image
                s = '%gx%g ' % img.shape[2:]  # print string
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                    # # Print results
                    # for c in det[:, -1].unique():
                    #     n = (det[:, -1] == c).sum()  # detections per class
                    #     s += f"{n} {self.class_names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # todo 在这里向json中append

        return pred