# -*- coding: utf-8 -*-
# TODO 添加模型运行需要导入的模块
import os
import torch
from collections import OrderedDict
import sys
from frcnn import FRCNN
sys.path.insert(0, os.path.dirname(__file__))
from PIL import Image
import os


# TODO 修改 MODEL_TYPE 为模型运行需要的引擎，此处支持pytorch和tensorflow，如需支持更多框架，请参考 https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0093.html "表1 BaseService类导入语句" 添加相关代码
MODEL_TYPE = 'pytorch'
ParentClass = None

try:
    if MODEL_TYPE == 'tensorflow':
        from model_service.tfserving_model_service import TfServingBaseService as ParentClass
    elif MODEL_TYPE == 'pytorch':
        from model_service.pytorch_model_service import PTServingBaseService as ParentClass
    else:
        ParentClass = object
except:
    ParentClass = object


class ModelClass(ParentClass):
    def __init__(self, model_name, model_path):
        """
        TODO 请在本方法中添加模型的构建和加载权重的过程，不同模型可根据不同的需要进行自定义修改
        :param model_name: 本参数必须保留，随意传入一个字符串值即可
        :param model_path: 模型所在的路径，比如 xxx/xxx.h5、xxx/xxx.pth，如果在ModelArts中运行，该参数会自动传入，不需要人工指定
        """
        self.model_name = model_name  # 本行代码必须保留，且无需修改
        self.model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
        self.class_names = {0: 'speed_unlimited', 1: 'red_stop', 2: 'speed_limited',
                            3: 'yellow_back', 4: 'green_go', 5: 'pedestrian_crossing'}

    def _preprocess(self, data):
        """
        本函数无需修改
        """
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                # 其实 我也估计 输入的图片 是直接使用image进行读取的
                img = Image.open(file_content)
                preprocessed_data[k] = img
        return preprocessed_data



    # 图片检测的推理过程：
    def _inference(self, data):
        """
        TODO 请在本函数中实现模型的推理过程，不同模型可根据不同的需要进行自定义修改
        """
        # 读取方式应该是Image进行读入的  也就是说 和我原先的代码读入方式 是完全相同的
        src_img = data['images']  # 本行代码必须保留，且无需修改
        #初始化检测器
        frcnn = FRCNN()
        """############# 以下为需要自定义修改的部分 #############"""
        # 这里对图片进行了必要的预处理  处理好的图片  可以直接输入网络进行分析
        # img, img_0 = image_np(src_img, img_size)
        # print('data', data, type(data))
        # 我对图片所进行的预处理——由于处理方式与原先predict是完全相同的 所以先尝试直接输入检测器





        with torch.no_grad():
            # detect()
            # det : [tensor([[272.09375,  80.31250, 341.90625, 180.68750,   0.96341,   2.00000]],
            # det = detect(self.model_path, img, img_0, img_size, device)
            # 先直接对 image 所读入的图片进行处理：
            # 现在是假设 读入的图片是image直接读入的格式
            det =  frcnn.detect_image(src_img)
            # 获取了 网络的输出结果
            result = det

        return result
