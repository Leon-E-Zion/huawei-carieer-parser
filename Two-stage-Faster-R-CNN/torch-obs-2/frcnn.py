import colorsys
import copy
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


from utils_ import DecodeBox, get_new_img_size


from collections import OrderedDict

# --------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#   如果出现shape不匹配
#   一定要注意训练时的NUM_CLASSES、
#   model_path和classes_path参数的修改
# --------------------------------------------#
class FRCNN(object):
    _defaults = {
        # "model_path": 'model_data/voc_weights_resnet.pth',
        # "classes_path": 'model_data/voc_classes.txt',
        "confidence": 0.5,
        "iou": 0.3,
        "backbone": "resnet50",
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化faster RCNN
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        # 定义类别
        self.classes_path = os.path.join(os.path.dirname(__file__), 'voc_classes.txt')
        self.class_names = self._get_class()

        self.model_path = os.path.join(os.path.dirname(__file__), 'resnet50.pth')
        self.generate()

        self.mean = torch.Tensor([0, 0, 0, 0]).repeat(self.num_classes + 1)[None]
        self.std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes + 1)[None]
        if self.cuda:
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()

        self.decodebox = DecodeBox(self.std, self.mean, self.num_classes)

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   载入模型  模型生成部分
    # ---------------------------------------------------#
    def generate(self):
        # -------------------------------#
        #   计算总的类的数量
        # -------------------------------#
        self.num_classes = len(self.class_names)

        # -------------------------------#
        #   载入模型与权值
        # -------------------------------#
        # 直接 一次性读取 模型的整个网络与所有参数
        # self.model = FasterRCNN(self.num_classes, "predict", backbone=self.backbone).eval()
        # print('Loading weights into state dict...')
        # 定义 所需要的GPU  定义 该网络使用 显卡进行运行
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # # # 模型加载
        # backbone = "resnet50"
        # NUM_CLASSES = 6
        # model = FasterRCNN(NUM_CLASSES, backbone=backbone)
        #
        # # self.model = torch.load(self.model_path, map_location=device)
        # # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model_dict = model.state_dict()
        # pretrained_dict = torch.load(self.model_path, map_location=device)
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)
        # # self.model.load_state_dict(state_dict)
        # # self.model = torch.load()
        # self.model = model
        # if self.cuda:
        #     # self.model = nn.DataParallel(self.model)
        #     self.model = self.model.cuda()
        # self.model = FasterRCNN(self.num_classes, "predict", backbone=self.backbone).eval()
        print('Loading weights into state dict...')
        device = torch.device('cpu')
        self.model = torch.load(self.model_path, map_location=device)
        # self.model.load_state_dict(state_dict)
        #
        # if self.cuda:
        #     # self.model = nn.DataParallel(self.model)
        #     self.model = self.model.cuda()
        # print('{} model, anchors, and classes loaded.'.format(self.model_path))
        print("model has been loaded")


    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        # -------------------------------------#
        #   转换成RGB图片，可以用于灰度图预测。
        # -------------------------------------#
        # 在 huawei_cloud项目中 所制作的提交文件 在此处传入的图片是image.open所直接打开的  未经任何操作便直接传入
        image = image.convert("RGB")
        # 对图片进行 预处理 操作
        image_shape = np.array(np.shape(image)[0:2])
        old_width, old_height = image_shape[1], image_shape[0]
        old_image = copy.deepcopy(image)

        # ---------------------------------------------------------#
        #   给原图像进行resize，resize到短边为600的大小上
        # ---------------------------------------------------------#
        width, height = get_new_img_size(old_width, old_height)
        image = image.resize([width, height], Image.BICUBIC)

        # -----------------------------------------------------------#
        #   图片预处理，归一化。
        # -----------------------------------------------------------#
        photo = np.transpose(np.array(image, dtype=np.float32) / 255, (2, 0, 1))


        # 图片预处理操作 完成 开始进行  网络推理
        with torch.no_grad():
            images = torch.from_numpy(np.asarray([photo]))
            # 图片是确实有输入网络的

            if self.cuda:
                images = images.cuda()
            # print(images)
            roi_cls_locs, roi_scores, rois, _ = self.model(images)
            # print(1)
            # print(roi_cls_locs)
            # print(roi_scores)
            # print(rois)
            # print(_)
            # print(1)
            # print(_)
            # -------------------------------------------------------------#
            #   利用classifier的预测结果对建议框进行解码，获得预测框
            # -------------------------------------------------------------#
            outputs = self.decodebox.forward(roi_cls_locs[0], roi_scores[0], rois, height=height, width=width,
                                             nms_iou=self.iou, score_thresh=self.confidence)
            # ---------------------------------------------------------#
            #   如果没有检测出物体，返回原图
            # ---------------------------------------------------------#
            # if len(outputs) == 0:
            #     return old_image
            # print(2)
            # print(outputs)
            # print(2)
            outputs = np.array(outputs)
            # print(outputs)
            bbox = outputs[:, :4]

            label = outputs[:, 4]
            conf = outputs[:, 5]
            # print(bbox)

            bbox[:, 0::2] = (bbox[:, 0::2]) / width * old_width
            bbox[:, 1::2] = (bbox[:, 1::2]) / height * old_height

        # 初始化容器
        detection_classes = []
        detection_scores = []
        detection_boxes = []

        result = OrderedDict()  # 本行代码必须保留，且无需修改

        output = []
        output.append(bbox)
        output.append(conf)
        output.append(label)

        bbox, conf, classes = output[0], output[1], output[2]
        for i in range(len(conf)):
            # 一个一个地 取出
            xyxy = bbox[i]
            detection_boxes.append([int(v) for v in xyxy])

            conf_=conf[i]
            detection_scores.append((float(conf_)))

            classe = classes[i]
            class_name = self.class_names[int(classe)]
            detection_classes.append(class_name)

        """############# 以上为需要自定义修改的部分，detection_classes、detection_scores、detection_boxes中不能含有np.ndarray数据类型 #############"""

        # result = OrderedDict()  # 本行代码必须保留，且无需修改
        result['detection_classes'] = detection_classes  # 如果返回的结果是物体检测格式，则本行代码必须保留，且无需修改
        result['detection_boxes'] = detection_boxes
        result['detection_scores'] = detection_scores  # 如果返回的结果是物体检测格式，则本行代码必须保留，且无需修改
        # result['detection_boxes'] = detection_boxes  # 如果返回的结果是物体检测格式，则本行代码必须保留，且无需修改

    # else:
    # result['detection_classes'] = []
    # result['detection_boxes'] = []
    # result['detection_scores'] = []


        return result
        # 以下 是有关绘制窗格图片的操作*****************************************************************

        # 定义 绘制框图时所需要用到的 字体字式
        # font = ImageFont.truetype(font='model_data/simhei.ttf',
        #                           size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        # thickness = max((np.shape(old_image)[0] + np.shape(old_image)[1]) // old_width * 2, 1)

        # # 获取一张 输入图片的复制图
        # image = old_image
        # # 开始迭代地 一个一个地将数据进行取出
        # for i, c in enumerate(label):
        #     # 取出预测类别，置信度 以及 框格的坐标信息
        #     predicted_class = self.class_names[int(c)]
        #     score = conf[i]
        #     left, top, right, bottom = bbox[i]
        #
        #     # 对 坐标信息 进行大致地偏移处理
        #     top = top - 5
        #     left = left - 5
        #     bottom = bottom + 5
        #     right = right + 5
        #
        #     # # 对 坐标信息 进行大致地 再 偏移处理
        #     top = max(0, np.floor(top + 0.5).astype('int32'))
        #     left = max(0, np.floor(left + 0.5).astype('int32'))
        #     bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
        #     right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))
        #
        #     # 画框框
        #     label = '{} {:.2f}'.format(predicted_class, score)
        #     draw = ImageDraw.Draw(image)
        #     label_size = draw.textsize(label, font)
        #     label = label.encode('utf-8')
        #     print(label, top, left, bottom, right)

            # 我们可以看到 在之下也再也没有对框格的数据进行再处理
            # 结合之前的注释 说明 框格数据在准备开始 可视化操作之前事实上已经还原到了 原始图片对应的尺寸
            # if top - label_size[1] >= 0:
            #     text_origin = np.array([left, top - label_size[1]])
            # else:
            #     text_origin = np.array([left, top + 1])
            #
            # for i in range(thickness):
            #     draw.rectangle(
            #         [left + i, top + i, right - i, bottom - i],
            #         outline=self.colors[int(c)])
            # draw.rectangle(
            #     [tuple(text_origin), tuple(text_origin + label_size)],
            #     fill=self.colors[int(c)])
            # draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            # del draw

        # return output

    # def get_FPS(self, image, test_interval):
    #     # -------------------------------------#
    #     #   转换成RGB图片，可以用于灰度图预测。
    #     # -------------------------------------#
    #     image = image.convert("RGB")
    #
    #     image_shape = np.array(np.shape(image)[0:2])
    #     old_width, old_height = image_shape[1], image_shape[0]
    #
    #     # ---------------------------------------------------------#
    #     #   给原图像进行resize，resize到短边为600的大小上
    #     # ---------------------------------------------------------#
    #     width, height = get_new_img_size(old_width, old_height)
    #     image = image.resize([width, height], Image.BICUBIC)
    #
    #     # -----------------------------------------------------------#
    #     #   图片预处理，归一化。
    #     # -----------------------------------------------------------#
    #     photo = np.transpose(np.array(image, dtype=np.float32) / 255, (2, 0, 1))
    #
    #     with torch.no_grad():
    #         images = torch.from_numpy(np.asarray([photo]))
    #         if self.cuda:
    #             images = images.cuda()
    #
    #         roi_cls_locs, roi_scores, rois, _ = self.model(images)
    #         # -------------------------------------------------------------#
    #         #   利用classifier的预测结果对建议框进行解码，获得预测框
    #         # -------------------------------------------------------------#
    #         outputs = self.decodebox.forward(roi_cls_locs[0], roi_scores[0], rois, height=height, width=width,
    #                                          nms_iou=self.iou, score_thresh=self.confidence)
    #         # ---------------------------------------------------------#
    #         #   如果没有检测出物体，返回原图
    #         # ---------------------------------------------------------#
    #         if len(outputs) > 0:
    #             outputs = np.array(outputs)
    #             bbox = outputs[:, :4]
    #             label = outputs[:, 4]
    #             conf = outputs[:, 5]
    #
    #             bbox[:, 0::2] = (bbox[:, 0::2]) / width * old_width
    #             bbox[:, 1::2] = (bbox[:, 1::2]) / height * old_height
    #
    #     t1 = time.time()
    #     for _ in range(test_interval):
    #         with torch.no_grad():
    #             roi_cls_locs, roi_scores, rois, _ = self.model(images)
    #             # -------------------------------------------------------------#
    #             #   利用classifier的预测结果对建议框进行解码，获得预测框
    #             # -------------------------------------------------------------#
    #             outputs = self.decodebox.forward(roi_cls_locs[0], roi_scores[0], rois, height=height, width=width,
    #                                              nms_iou=self.iou, score_thresh=self.confidence)
    #             # ---------------------------------------------------------#
    #             #   如果没有检测出物体，返回原图
    #             # ---------------------------------------------------------#
    #             if len(outputs) > 0:
    #                 outputs = np.array(outputs)
    #                 bbox = outputs[:, :4]
    #                 label = outputs[:, 4]
    #                 conf = outputs[:, 5]
    #
    #                 bbox[:, 0::2] = (bbox[:, 0::2]) / width * old_width
    #                 bbox[:, 1::2] = (bbox[:, 1::2]) / height * old_height
    #
    #     t2 = time.time()
    #     tact_time = (t2 - t1) / test_interval
    #     return tact_time
