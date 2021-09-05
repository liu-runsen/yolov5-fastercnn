import json
import os
from tqdm import tqdm


def json_to_yolo(yolo_save_path,anno_file):

    if not os.path.exists(yolo_save_path):
        os.makedirs(yolo_save_path)

    with open(anno_file, 'r') as f:
        train_anno = json.load(f)
    imageid2annos = dict()
    for anno in train_anno["annotations"]:
        imageid = anno["image_id"]
        if imageid not in imageid2annos:
            imageid2annos[imageid] = []
        imageid2annos[imageid].append(anno)

    for image in tqdm(train_anno["images"]):
        imageid = image["id"]
        imagename = image["file_name"]
        image_h = image["height"]
        image_w = image["width"]
        dh = 1 / image_h
        dw = 1 / image_w
        with open(yolo_save_path + imagename.split(".")[0] + ".txt", "w") as f:
            for anno in imageid2annos[imageid]:
                xmin, ymin, w, h = anno["bbox"]
                xmax = xmin + w
                ymax = ymin + h
                x_center = (xmin + xmax) / 2.0
                x_center = x_center * dw
                y_center = (ymin + ymax) / 2.0
                y_center = y_center * dh
                w *= dw
                h *= dh
                mystring = "0" + " " + str(round(x_center,7)) + " " +  str(round(y_center,7)) + " " +  str(round(w,7)) + " " +  str(round(h,7))
                f.write(mystring)
                f.write("\n")

if __name__ == '__main__':
    yolo_save_path = 'mini_airplane/labels/'
    anno_file = "mini_airplane/annotations/train.json"

    json_to_yolo(yolo_save_path,anno_file)

    yolo_save_path = 'mini_airplane/labels/'
    anno_file = "mini_airplane/annotations/val.json"

    json_to_yolo(yolo_save_path, anno_file)
