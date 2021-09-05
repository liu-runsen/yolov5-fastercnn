# coding:utf-8

'''
@Author：Runsen
'''

# coding:utf-8

import os
import random
import argparse

parser = argparse.ArgumentParser()
#xml文件的地址，根据自己的数据进行修改 xml一般存放在Annotations下
parser.add_argument('--xml_path', default='./mini_airplane/Annotation/', type=str, help='input xml label path')


#数据集的划分，地址选择自己数据下的ImageSets/Main
parser.add_argument('--txt_path', default='./mini_airplane/ImageSets/Main/', type=str, help='output txt label path')

opt = parser.parse_args()

train_percent = 0.85
xmlfilepath = opt.xml_path
txtsavepath = opt.txt_path
total_xml = os.listdir(xmlfilepath)
if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

num = len(total_xml)
list_index = range(num)
tr = int(num * train_percent)
trainval = random.sample(list_index, num)
train = random.sample(trainval, tr)

file_test = open(txtsavepath + 'test.txt', 'w')
file_train = open(txtsavepath + 'train.txt', 'w')
file_val = open(txtsavepath + 'val.txt', 'w')

for i in list_index:
    name = total_xml[i][:-4] + ".jpg" + '\n'
    if i in trainval:

        if i in train:
            file_train.write("/home/lrs/demo/mini_airplane/images/" + name   )
        else:
            file_val.write("/home/lrs/demo/mini_airplane/images/" + name )
    else:
        file_test.write("/home/lrs/demo/mini_airplane/images/" + name  )

file_train.close()
file_val.close()
file_test.close()