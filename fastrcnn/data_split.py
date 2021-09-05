# coding:utf-8

'''
@Author：Runsen
'''
import json
import shutil

data_root = '../mini_airplane/'

with open(data_root+"annotations/instances_val2014.json", 'r') as f:
    annos = json.load(f)

images = annos["images"]
imageid2image = {}

for image in images:
    imageid2image[image['id']]=image

categories = annos['categories']
cateid2name={}
for cate in categories:
    cateid2name[cate['id']] = cate['name']

image_final=[]
annos_final=[]
categories_final=[{
"id":1,
"name": "airplane"}]
image_id=1
anno_id=1
imgid2airplane={}
for anno in annos['annotations']:
    if cateid2name[anno['category_id']] == 'airplane':
        image_id_t = anno['image_id']
        if image_id_t not in imgid2airplane:
            imgid2airplane[image_id_t] = []
        imgid2airplane[image_id_t].append(anno)

for imgid, annos in imgid2airplane.items():
    #print(image_id)
    imagename = imageid2image[imgid]['file_name']
    shutil.copy(data_root + 'val2014/' + imagename, 'mini_airplane/' + imagename)
    image =  imageid2image[imgid]
    image['id'] = image_id
    image_final.append(image)
    for anno in annos:
        anno["id"] = anno_id
        anno["image_id"] = image_id
        anno["category_id"] = 1
        anno_id+=1
        annos_final.append(anno)
    image_id+=1
    if image_id >=100:
        break

instance = {"images": image_final,
            "annotations":annos_final,
            "categories": categories_final}

with open('mini_airplane_train.json', 'w') as f:
    json.dump(instance, f, indent=1)
