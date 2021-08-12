#*************************************************************************************************用于定义 框架 推理 的相关必要代码

import os
# 用于 加载安装  XXX.whl文件


import sys
from model_service.pytorch_model_service import PTServingBaseService

# 加载工具包
from utils_net import *

# 定义运行装置 --CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义各种超参数
code_url = os.path.dirname(__file__)
sys.path.insert(0, code_url)
_IMG_SIZE = 640
_CONF_THRES = 0.4
_IOU_THRES = 0.5


class YOLOv5(PTServingBaseService):
    def __init__(self, model_name, model_path):
        # 初始化各种 信息
        self.model_name = model_name
        self.model_path = model_path
        self.model =  load_weights_(model_path, map_location=device)
        self.half = (device.type != 'cpu')

        # 模型  降精度  加速推理
        if self.half:
            self.model.half()

        # 获取各类标签
        self.labels = self.model.module.name \
            if hasattr(self.model, 'module') else self.model.names


    # 定义 推理过程
    def _preprocess(self, data):
        data_list = []

        # 对输入 图片的数据 进行逐一处理
        for _, v in data.items():
            for _, file_content in v.items():
                file_content = file_content.read()

                # 转化成 输入网络的图片
                img = cv2.imdecode(np.frombuffer(file_content, np.uint8), cv2.IMREAD_COLOR)
                assert img is not None

                # 获取输入图片的尺寸
                self.img0_size = img.shape

                # 转化为 指定的大小  输入网络的大小
                img = letterbox(img, new_shape=_IMG_SIZE)[0]

                # BGR to RGB
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)

                # img为输入神经网络的图片，img0为原图
                self.img_size = img.shape[1:]

                # 对图片进行预处理
                img = torch.from_numpy(img).to(device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0

                # 标准化输入图片的格式
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # 将数据 载入图片——是输入网络的数据集合
                data_list.append(img)

        return data_list

    # 定义推理过程  获取网络的输出
    def _inference(self, data):
        with torch.no_grad():
            data = self.model(data[0])
        return data

    # 对网络输出进行 处理，并按照赛题要求格式返回需要的信息
    def _postprocess(self, data):

        # 定义 所需要输出的数据的容器
        result_return = dict()

        # 对网络输出进行非极大值抑制处理
        pred = non_max_suppression(data[0], _CONF_THRES, _IOU_THRES)

        # 对最终的网络输出进行格式上调整
        if pred[0] is not None:

            # 预测值 处理_0：整体处理
            picked_boxes = scale_coords(self.img_size, pred[0][:, :4], self.img0_size).round().to(torch.device('cpu')).detach().numpy()
            pred = pred[0].to(torch.device('cpu')).detach().numpy()

            # 预测值 处理_1：分类化处理
            picked_boxes = picked_boxes[:, [1, 0, 3, 2]]
            picked_classes = self.convert_labels(pred[:, 5])
            picked_score = pred[:, 4]

            # 数据装载过程
            result_return['detection_classes'] = picked_classes
            result_return['detection_boxes'] = picked_boxes.tolist()
            result_return['detection_scores'] = picked_score.tolist()

        # 如果 输入图片中 没有需要识别的目标对象
        else:
            result_return['detection_classes'] = []
            result_return['detection_boxes'] = []
            result_return['detection_scores'] = []

        # 返回需要输出的信息
        return result_return

    # 将网络输出的类别序号  转化为 标签的形式进行返回
    def convert_labels(self, label_list):
        if isinstance(label_list, np.ndarray):
            label_list = label_list.tolist()
        label_names = [self.labels[int(index)] for index in label_list]
        return label_names
