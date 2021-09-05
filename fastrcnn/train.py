# coding:utf-8

'''
@Author：Runsen
'''

# 这个Averager用于统计训练过程中的loss，便于打印
import json

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from fastrcnn.dataset import train_data_loader, valid_data_loader
from fastrcnn.model import model


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


# collate_fn函数用于将一个batch的数据转换为tuple格式

# 使用上面的AirplaneDetDataset来定义训练和验证的数据类


params = [p for p in model.parameters() if p.requires_grad]
# 建立sgd 优化器
optimizer = torch.optim.SGD(params, lr=0.0025, momentum=0.9, weight_decay=0.0005)
lr_scheduler = None
# 一共训练100个epoch
num_epochs = 100  # 100

# resnet50模型过大，内存不足
loss_hist = Averager()
itr = 1
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 按照每一个epoch循环训练
for epoch in range(num_epochs):
    loss_hist.reset()
    # 从data loader 中采样每个batch数据用于训练
    for images, targets, _ in train_data_loader:

        # 将image 和 label 挪到GPU中
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 数据进入模型并反向传播得到loss
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        loss_hist.send(loss_value)

        # 更新模型参数
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if itr % 10 == 0:
            print(f"Iteration #{itr} loss: {loss_value}")

        itr += 1
        if itr > 10: break;

    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

    print(f"Epoch #{epoch} loss: {loss_hist.value}")



itr = 1
for images, _, imgname in iter(valid_data_loader):
    images = list(img.to(device) for img in images)
    sample = images[0].permute(1, 2, 0).cpu().numpy()
    model.eval()
    cpu_device = torch.device("cpu")
    with torch.no_grad():
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
    #     fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    boxes = outputs[0]['boxes'].data.cpu().numpy()
    scores = outputs[0]['scores'].data.cpu().numpy()

    boxes = boxes[scores >= 0.8].astype(np.int32)
    for box in boxes:
        cv2.rectangle(np.ascontiguousarray(sample),
                      (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                      (220, 0, 0), 3)

    plt.imshow(sample)
    plt.savefig('../results/val_vis/' + str(imgname[0]))
    plt.show()
    itr += 1
    if itr > 30:
        break

# 首先预测每张验证集的图片，将每张图片的结果保存在字典imgname2bboxes中
imgname2bboxes = dict()
for images, _, imgname in iter(valid_data_loader):
    imgname = imgname[0]
    imgname2bboxes[imgname] = []
    images = list(img.to(device) for img in images)
    sample = images[0].permute(1, 2, 0).cpu().numpy()
    cpu_device = torch.device("cpu")
    with torch.no_grad():
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
    boxes = outputs[0]['boxes'].data.cpu().numpy()
    scores = outputs[0]['scores'].data.cpu().numpy()

    boxes = boxes[scores >= 0.05].astype(np.int32)
    scores = scores[scores >= 0.05]
    for box, score in zip(boxes, scores):
        imgname2bboxes[imgname].append([box[0], box[1], box[2], box[3], score])

# 将imgname2bboxes字典的内容转换为COCO标准格式并保存下来，用于之后的计算mAP

with open("../mini_airplane/annotations/val.json", "r") as f:
    val_labels = json.load(f)
val_images = val_labels["images"]
imagename2id = dict()
for img in val_images:
    imagename2id[img["file_name"]] = img["id"]
val_annotations = []
anno_id = 1
for imgname, bboxes in imgname2bboxes.items():
    for bbox in bboxes:
        box = list([int(bbox[0]), int(bbox[1]), int(bbox[2]) - int(bbox[0]) + 1, int(bbox[3]) - int(bbox[1]) + 1])
        score = float(bbox[-1])
        imgid = imagename2id[imgname]
        val_annotations.append({
            "id": anno_id,
            "bbox": box,
            "score": score,
            "image_id": imagename2id[imgname],
            "category_id": 1
        })
        anno_id += 1

with open("../results/val_results.json", "w") as f:
    json.dump(val_annotations, f, indent=2)

torch.save(model.state_dict(), '../models/fasterrcnn_resnet50_fpn.pth')
