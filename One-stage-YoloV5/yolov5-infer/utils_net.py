import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
import time



# 模型容器  装载输出
class Output_box(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Output_box, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])

        y = torch.stack(y).mean(0)
        return y, None

# 批量  加载模型
def load_weights_(weights, map_location=None):
    model = Output_box()
    for w in weights if isinstance(weights, list) else [weights]:
        model.append(torch.load(w, map_location=map_location)['model'].float().fuse().eval())  # load FP32 model

    if len(model) == 1:
        return model[-1]
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model

# 将坐标  从 中心点宽高的形式转化到 四个点 的表示形式
def xywh2xyxy(x):

    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


# 计算 坐标表达式的IoU
def box_iou(box1, box2):
    def box_area(box):
        r_box = (box[2] - box[0]) * (box[3] - box[1])
        return   r_box

    ar_0 = box_area(box1.T)
    ar_1 = box_area(box2.T)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)

    b_IoU = inter / (ar_0[:, None] + ar_1 - inter)

    return b_IoU



# 极大值抑制算法
def non_max_suppression(prediction, conf_thres=0.1,
                        iou_thres=0.6, merge=False, classes=None, agnostic=False):

    if prediction.dtype is torch.float16:
        prediction = prediction.float()

    nc = prediction[0].shape[1] - 5
    xc = prediction[..., 4] > conf_thres


    min_wh, max_wh = 2, 4096
    max_det = 300
    time_limit = 10.0
    redundant = True
    multi_label = nc > 1

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):

        x = x[xc[xi]]


        if not x.shape[0]:
            continue

        # 计算置信度
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # 将 输入坐标从中心点坐标和宽高的形式 转化到 四点坐标的形式
        box = xywh2xyxy(x[:, :4])


        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # 对输出的选择作用
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # 类别处理
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue



        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


# 对输入图片进行 低失真的尺寸 调整
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True,
              scaleFill=False, scaleup=True):

    # 获取当前的图片尺寸
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 裁剪尺寸
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # 计算填充块的大小
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

# 图片尺寸的调整操作
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    # x方向上的padding
    coords[:, [0, 2]] -= pad[0]
    # y方向上的padding
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain


    # bbox的尺寸调整
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2

    # 返回需要的输出
    return coords
