#!/usr/bin/env python
# coding=utf-8
'''
将生成patch聚合为癌症区域并嵌入到normal切块中
'''

import os
import cv2
import random
import tools
import numpy as np
import makevocxml

inputGenedir = '../../mydata/dcgan/dcgan_micro_512/'
inputNordir = '../../mydata/dcgan/normal_part/'
outputdir = '../../mydata/dcgan/virtual_dataset/'
outputbboxdir = '../../mydata/dcgan/virtual_dataset_bbox/'
outputxmldir = '../../mydata/dcgan/virtual_dataset_xml/'
outputmaskdir = '../../mydata/dcgan/virtual_dataset_mask/'
tools.mkdir(outputdir)
tools.mkdir(outputbboxdir)
tools.mkdir(outputxmldir)
tools.mkdir(outputmaskdir)


def make_region(gene_list, w_num, h_num):
    vstack = []
    for i in range(h_num):
        hstack = []
        for j in range(w_num):
            img = cv2.imread(inputGenedir+gene_list[i*w_num+j])
            hstack.append(img)
        image = np.concatenate(hstack, axis=1)
        vstack.append(image)
    img_region = np.concatenate(vstack)
    return img_region


def inpainting(gene_all_list, nor_all_list, gene_size, nor_size, number):
    x = random.randint(0, nor_size-gene_size*2)
    y = random.randint(0, nor_size-gene_size*2)
    w_num_max = (nor_size-x)/gene_size
    w_num = random.randint(1, w_num_max)
    w = w_num * gene_size
    h_num_max = (nor_size-y)/gene_size
    h_num = random.randint(1, h_num_max)
    h = h_num * gene_size

    gene_list = random.sample(gene_all_list, w_num*h_num)
    nor_list = random.sample(nor_all_list, 1)
    img_region = make_region(gene_list, w_num, h_num)
    img_part = cv2.imread(inputNordir+nor_list[0])
    img_part[y:y+h, x:x+w] = img_region
    number += 1
    cv2.imwrite(outputdir+str(number)+'.jpg', img_part)

    img_part_bbox = cv2.rectangle(img_part, (x,y), (x+w,y+h), (0,255,0), 8)
    cv2.imwrite(outputbboxdir+str(number)+'.jpg', img_part_bbox)

    bbox_label = []
    bbox_label.append([x,y,x+w,y+h,w*h,1, 'Tumor'])
    makevocxml.makexml(outputxmldir, number, img_part.shape, bbox_label)

    mask = np.zeros((nor_size,nor_size,1), np.uint8)
    mask_region = np.zeros((h,w,1), np.uint8)
    mask_region[:] = 255
    mask[y:y+h, x:x+w] = mask_region
    cv2.imwrite(outputmaskdir+str(number)+'.jpg', mask)

    return number


if __name__ == "__main__":
    gene_size = 64
    nor_size = 512
    number = 0
    total = 20
    gene_all_list = []
    nor_all_list = []
    for parents, dirnames, filenames in os.walk(inputGenedir):
        for f in filenames:
            gene_all_list.append(f)
    for parents, dirnames, filenames in os.walk(inputNordir):
        for f in filenames:
            nor_all_list.append(f)
    for i in range(total):
        number = inpainting(gene_all_list, nor_all_list, gene_size, nor_size, number)
