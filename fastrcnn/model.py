# coding:utf-8

'''
@Author：Runsen
'''

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor





def get_object_detection_model(num_classes):
    # 载入一个在COCO数据集上预训练好的faster rcnn 模型，backbone为resnet50，neck网络使用fpn
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # 获取最后分类的head的输入特征维度
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 将最后分类的head从原始的COCO输出81类替换为我们现在输入的num_classes类，注意这里的num_classes=实际的类别数量+1，1代表背景
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# 构建我们的基于faster rcnn的飞机检测模型，类别数量如上所讲述
# 1 class (person) + background
model = get_object_detection_model(2).to(device)