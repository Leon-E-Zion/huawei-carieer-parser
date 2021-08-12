# Copyright 2019 ModelArts Authors from Huawei Cloud. All Rights Reserved.
# https://www.huaweicloud.com/product/modelarts.html
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import json
import logging
import os
import numpy as np
from PIL import Image
from model_service.pytorch_model_service import PTServingBaseService
import time
from utils.utils import DecodeBox, get_new_img_size
import cv2
import numpy as np
from PIL import Image

from frcnn import FRCNN

current_path = os.path.split(os.path.abspath(__file__))[0]

EPS = np.finfo(float).eps
# with open(os.path.join(current_path, 'index'), 'r') as f:
#   index_map = json.loads(f.read())
# class_names = index_map
class_names = ["red_stop","green_go","yellow_back","pedestrian_crossing","speed_limited","speed_unlimited"]

class_num = len(class_names)

# 需要调整的  输入图片的参数——最好去对照ssd的文件看看
net_h = 600
net_w = 600
# 这里也需要根据自己的模型的超参数去调整
conf_threshold = 0.3
iou_threshold = 0.45
_scale_factors = [10.0, 10.0, 5.0, 5.0]
channel_means = [123.68, 116.779, 103.939]


def _sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def get_center_coordinates_and_sizes(bbox):
    ymin, xmin, ymax, xmax = np.transpose(bbox)
    width = xmax - xmin
    height = ymax - ymin
    ycenter = ymin + height / 2.
    xcenter = xmin + width / 2.
    return [ycenter, xcenter, height, width]


def combined_static_and_dynamic_shape(tensor):
  static_tensor_shape = list(tensor.shape)
  dynamic_tensor_shape = tensor.shape
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape


def batch_decode(box_encodings, anchors, class_predictions, ori_h, ori_w):
    combined_shape = combined_static_and_dynamic_shape(box_encodings)
    batch_size = combined_shape[0]
    tiled_anchor_boxes = np.tile(
        np.expand_dims(anchors, 0), [batch_size, 1, 1])
    tiled_anchors_boxlist = np.reshape(tiled_anchor_boxes, [-1, 4])
    decoded_boxes = decode_bbox_frcnn(
        np.reshape(box_encodings, [-1, 4]),tiled_anchors_boxlist, ori_h, ori_w)
    decoded_boxes = np.reshape(decoded_boxes, np.stack(
        [combined_shape[0], combined_shape[1], 4]))
    return decoded_boxes


def overlap(x1, x2, x3, x4):
    left = max(x1, x3)
    right = min(x2, x4)
    return right - left


def cal_iou(box, truth):
    w = overlap(box[0], box[2], truth[0], truth[2])
    h = overlap(box[1], box[3], truth[1], truth[3])
    if w <= 0 or h <= 0:
        return 0
    inter_area = w * h
    union_area = (box[2] - box[0]) * (box[3] - box[1]) + (truth[2] - truth[0]) * (truth[3] - truth[1]) - inter_area
    return inter_area * 1.0 / union_area

# 极大值抑制 输入所有框框 与 阈值, 输出对应的 处理后的输出 ， 输出一个列表的  筛选过的 框
def apply_nms(all_boxes, thres):
    res = []

    for cls in range(class_num):
        cls_bboxes = all_boxes[cls]
        sorted_boxes = sorted(cls_bboxes, key=lambda d: d[5])[::-1]

        p = dict()
        for i in range(len(sorted_boxes)):
            if i in p:
                continue

            truth = sorted_boxes[i]
            for j in range(i + 1, len(sorted_boxes)):
                if j in p:
                    continue
                box = sorted_boxes[j]
                iou = cal_iou(box, truth)
                if iou >= thres:
                    p[j] = 1

        for i in range(len(sorted_boxes)):
            if i not in p:
                res.append(sorted_boxes[i])
    return res

# 把 输出的一个框子 变成可视化的形式 的一个框子
def decode_bbox_frcnn(conv_output, anchors, ori_h, ori_w):
    ycenter_a, xcenter_a, ha, wa = get_center_coordinates_and_sizes(anchors)
    ty, tx, th, tw = conv_output.transpose()
    ty /= _scale_factors[0]
    tx /= _scale_factors[1]
    th /= _scale_factors[2]
    tw /= _scale_factors[3]

    w = np.exp(tw) * wa
    h = np.exp(th) * ha

    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a

    scale_h = 1
    scale_w = 1
    ymin = np.maximum((ycenter - h / 2.) * scale_h * ori_h, 0)
    xmin = np.maximum((xcenter - w / 2.) * scale_w * ori_w, 0)
    ymax = np.minimum((ycenter + h / 2.) * scale_h * ori_h, ori_h)
    xmax = np.minimum((xcenter + w / 2.) * scale_w * ori_w, ori_w)

    bbox_list = np.transpose(np.stack([ymin, xmin, ymax, xmax]))

    return bbox_list

# 后处理方法 ，将网络的输出转换为API接口 输入框格，盒子_编码es，类别预测信息es，原图的尺寸
def postprocess(anchors, box_encodings, class_predictions, ori_h, ori_w):








    detection_boxes = batch_decode(box_encodings, anchors, class_predictions, ori_h, ori_w)
    detection_scores_with_background = _sigmoid(class_predictions)
    detection_scores = detection_scores_with_background[:, :, 1:]
    detection_score = np.expand_dims(np.max(detection_scores[:, :, 0:], axis=-1), 2)
    detection_cls = np.expand_dims(np.argmax(detection_scores[:, :, 0:], axis=-1), 2)

    preds = np.concatenate([detection_boxes, detection_score, detection_cls], axis=2)
    res_list = []
    for pred in preds:
        pred = pred[pred[:, 4] >= conf_threshold]
        all_boxes = [[] for ix in range(class_num)]
        for ix in range(pred.shape[0]):
            box = [int(pred[ix, iy]) for iy in range(4)]
            box.append(int(pred[ix, 5]))
            box.append(pred[ix, 4])
            all_boxes[box[4] - 1].append(box)

        res = apply_nms(all_boxes, iou_threshold)
        res_list.append(res)
    return res_list


def get_result(result, img_h, img_w):
    anchors = result["anchors"]
    box_encodings = result["box_encodings"]
    class_predictions_with_background = result["class_predictions_with_background"]
    detection_dict = postprocess(anchors, box_encodings,
                                 class_predictions_with_background,
                                 img_h, img_w)

    return detection_dict

def get_new_img_size(width, height, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_width, resized_height

# 前处理阶段
class object_detection_service(PTServingBaseService):
  def _preprocess(self, data):
    preprocessed_data = {}

    self.class_names = class_names
    self.num_classes = 6
    self.iou =  0.3,
    self.confidence = 0.5,
    # ______________________________________________________________________这里可以这么使用torch吗
    self.mean = torch.Tensor([0, 0, 0, 0]).repeat(self.num_classes + 1)[None]
    self.std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes + 1)[None]

    self.decodebox = DecodeBox(self.std, self.mean, self.num_classes)
    for k, v in data.items():
      for file_name, file_content in v.items():
        image = Image.open(file_content)
        if image.mode != 'RGB':
          logging.warning('Input image is not RGB mode, it will cost time to convert to RGB!'
                          'Input RGB mode will reduce infer time.')
          image = image.convert('RGB')

        image = image.resize((net_h, net_w), Image.ANTIALIAS)
        image = np.asarray(image, dtype=np.float32)
        image = image - channel_means
        image = image[np.newaxis, :, :, :]
        self.img_w, self.img_h = image.size
        preprocessed_data[k] = image
    return preprocessed_data

  def _postprocess(self, data):
      # data 是网络的输出
      # result = get_result(data, self.img_h, self.img_w)

      response = {
          'detection_classes': [],
          'detection_boxes': [],
          'detection_scores': []
      }
      # for i in range(len(result[0])):
      #     ymin = result[0][i][0]
      #     xmin = result[0][i][1]
      #     ymax = result[0][i][2]
      #     xmax = result[0][i][3]
      #     score = result[0][i][5]
      #     label = result[0][i][4]
      old_width, old_height = self.img_w, self.img_h
      width, height = get_new_img_size(old_width, old_height)

      roi_cls_locs, roi_scores, rois, _ = data
      outputs = self.decodebox.forward(roi_cls_locs[0], roi_scores[0], rois, height=height, width=width,
                                       nms_iou=self.iou, score_thresh=self.confidence)

      outputs = np.array(outputs)
      bbox = outputs[:, :4]
      label = outputs[:, 4]
      conf = outputs[:, 5]


      bbox[:, 0::2] = (bbox[:, 0::2]) / width * old_width
      bbox[:, 1::2] = (bbox[:, 1::2]) / height * old_height

      for i, c in enumerate(label):

          xmin ,ymin ,xmax , ymax = bbox[i]

          score = conf[i]
          label = label[i]

          response['detection_classes'].append(
              class_names[int(label)])
          response['detection_boxes'].append([
              str(ymin), str(xmin),
              str(ymax), str(xmax)
          ])
          response['detection_scores'].append(str(score))
      return response
