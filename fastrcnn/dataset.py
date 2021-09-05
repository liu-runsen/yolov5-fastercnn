# coding:utf-8

'''
@Author：Runsen
'''
import json
import os
import shutil
import cv2
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader



class AirplaneDetDataset(torch.utils.data.Dataset):
    def __init__(self, root, anno_file, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = []
        # 从label文件中载入所有的图片，并且预先生成图片名称和图片label的对应，便于__getitem__使用
        with open(os.path.join(root, "annotations", anno_file), "r") as f:
            labels = json.load(f)
        self.imagename2id = {}
        self.imageid2anno = {}
        for image in labels["images"]:
            self.imgs.append(image["file_name"])
            self.imagename2id[image["file_name"]] = image["id"]
        for anno in labels["annotations"]:
            if anno["image_id"] not in self.imageid2anno:
                self.imageid2anno[anno["image_id"]] = []
            self.imageid2anno[anno["image_id"]].append(anno["bbox"])

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img_path = os.path.abspath(img_path)
        # 载入图片并转换为pytorch所需格式
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#         img = io.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = img / 255.0
        img = img.transpose(2, 0, 1)
        img = torch.tensor(img)
        img_id = self.imagename2id[self.imgs[idx]]
        boxes = []
        # 读取该图片对应的bbox label
        for bbox in self.imageid2anno[img_id]:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[0] + bbox[2]
            ymax = bbox[1] + bbox[3]
            boxes.append([xmin, ymin, xmax, ymax])
        num_objs = len(self.imageid2anno[img_id])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # 只有飞机这一类，所以所有类别label都给1
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # 这里我们假设所有的目标label都是不互相严重遮挡的
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        # 最终我们输出的label是一个dict的格式，里面包括以下的字段用于训练
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        # 最终返回图片矩阵，label以及图片名
        return img, target, self.imgs[idx]

    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = AirplaneDetDataset('../mini_airplane', 'train.json')
valid_dataset = AirplaneDetDataset('../mini_airplane', 'val.json')
# 定义训练data_loader
train_data_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)
# 定义验证data_loader
valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

# if __name__ == '__main__':
#
#
#     dataset = AirplaneDetDataset('../mini_airplane', 'train.json')
#     print(dataset[0])
