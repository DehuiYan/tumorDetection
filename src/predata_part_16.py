#!/usr/bin/env python
# coding=utf-8


import openslide
from openslide import OpenSlide
from PIL import Image
import cv2
import numpy as np
import sys
import makevocxml
import time
import writetxt
import tools
import math


originDir = '../../mydata/camelyon16/TrainingData/'
inputDir = '../../mydata/VOCdevkit/VOC16_' + time.ctime() + '/'
originMaskDir = originDir+'Ground_Truth/Mask/'
originTrainDir = originDir+'Train_Tumor/'
originXmlDir = originDir+'Ground_Truth/XML/'

inputPartDir = inputDir+'JPEGImages/'
inputPartBboxDir = inputDir+'JPEGImages_bbox/'
inputXmlDir = inputDir+'Annotations/'
inputInfoDir = inputDir+'info/'
inputMaskDir = inputDir+'Mask/'
inputSetDir = inputDir+'ImageSets/'
inputMainDir = inputSetDir+'Main/'
tools.mkdir(inputDir)
tools.mkdir(inputPartDir)
tools.mkdir(inputPartBboxDir)
tools.mkdir(inputXmlDir)
tools.mkdir(inputInfoDir)
tools.mkdir(inputMaskDir)
tools.mkdir(inputSetDir)
tools.mkdir(inputMainDir)

filename = ''
LEVEL0 = 0
LEVEL3 = 3
LEVEL6 = 6

def region_newlevel(x,y,w,h,l1,l2,ds1,ds2):
    rate_x = ds1[0] / ds2[0]
    rate_y = ds1[1] / ds2[1]
    xx = int(x*ds1[0])
    yy = int(y*ds1[1])
    ww = int(w*rate_x)
    hh = int(h*rate_y)
    return xx,yy,ww,hh

def get_tissue(osr, dstime = 3, kernel_para = 15):
    img = osr.read_region((0,0), LEVEL3,  osr.level_dimensions[LEVEL3])
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for i in range(dstime):
        img = cv2.pyrDown(img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    blur = cv2.GaussianBlur(s,(5,5),0)
    ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((kernel_para, kernel_para),np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    #show threshold segment result
    cv2.imwrite(inputInfoDir+filename+'_'+'contours_level'+str(LEVEL3)+'_ds'+str(dstime)+'.jpg', th)
    for i in range(dstime):
        th = cv2.pyrUp(th)
        img = cv2.pyrUp(img)
    image, contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, img

#------------use for making data----------------
def intersection(rect1, rect2):
    rect = [0]*4
    rect[0] = max(rect1[0], rect2[0])
    rect[1] = max(rect1[1], rect2[1])
    rect[2] = min(rect1[0]+rect1[2], rect2[0]+rect2[2])
    rect[3] = min(rect1[1]+rect1[3], rect2[1]+rect2[3])
    if rect[0] < rect[2] and rect[1] < rect[3]:
        return (rect[2]-rect[0])*(rect[3]-rect[1])
    else:
        return 0


def make_data_and_label(osr, xx, yy, ww, hh, number, downsamples, part_dstime, mask, xml_bbox, part_size):
    x,y,w,h = region_newlevel(xx,yy, ww,hh,LEVEL3,LEVEL0, downsamples[LEVEL3], downsamples[LEVEL0])
    w = part_size
    h = part_size
    flag = False
    for i in xml_bbox:
        if intersection([x,y,w,h], [i[0],i[1],i[2],i[3]]) != 0:
            flag = True
            break
    if flag == False:
        return number
    part_mask = mask.read_region((x,y), LEVEL0, (w,h))
    part_mask = cv2.cvtColor(np.array(part_mask), cv2.COLOR_RGB2GRAY)
    image, contours, hierarchy = cv2.findContours(part_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return number

    bbox_label_this = []
    part = osr.read_region((x,y), LEVEL0, (w,h))
    part = cv2.cvtColor(np.array(part), cv2.COLOR_RGB2BGR)
    part_bbox = part.copy()
    flag = False
    for i, cnt in enumerate(contours):
        mask_x, mask_y, mask_w, mask_h = cv2.boundingRect(cnt)
        #print mask_x, mask_y, mask_w, mask_h
        area = cv2.contourArea(cnt)
        extent = float(area)/(mask_w*mask_h)
        if area < 16 * 16:
            continue
        if extent < 0.3:
            continue
        flag = True
        #if Major_axis < 500 and area < 100 * 100:
        #if math.sqrt(mask_w**2 + mask_h**2) < 850:
        #    variety = 'ITC'
        #    color = (255,0,255)
        #else:
        variety = 'Tumor'
        color = (0,255,0)
        bbox_label_this.append([mask_x, mask_y, mask_x+mask_w, mask_y+mask_h, area, extent, variety])
        part_bbox = cv2.rectangle(part_bbox, (mask_x, mask_y), (mask_x+mask_w, mask_y+mask_h), color, 32)
    if flag == False:
        return number
    number += 1
    for i in range(part_dstime):
        part = cv2.pyrDown(part)
        part_bbox = cv2.pyrDown(part_bbox)
        part_mask = cv2.pyrDown(part_mask)
        for j in bbox_label_this:
            j[0] /= 2
            j[1] /= 2
            j[2] /= 2
            j[3] /= 2
            j[4] /= 4
            j[5] /= 1 

    c = str(number)
    ze = 6 - len(c)
    cv2.imwrite(inputPartDir+'0'*ze + c +'.jpg', part)
    cv2.imwrite(inputPartBboxDir+'0'*ze+c+'.jpg', part_bbox)
    cv2.imwrite(inputMaskDir+'0'*ze + c +'.jpg', part_mask)
    makevocxml.makexml(inputXmlDir, number, part.shape, bbox_label_this)
    return number


def just_do_it(osr, number, begin_level_dstime, downsamples, part_size, part_step, part_dstime, mask, xml_bbox):
    contours, img = get_tissue(osr, begin_level_dstime)
    contours_bbox = []
    for i, cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        contours_bbox.append([x,y,w,h])
    for i in contours_bbox:
        x = i[0]
        y = i[1]
        w = i[2]
        h = i[3]
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),16)
        #print x, y, w, h
        part_w = part_size
        part_h = part_size

        for j in range((h-part_size)/part_step + 1):
            for k in range((w-part_size)/part_step + 1):
                part_x = x + part_step*k
                part_y = y + part_step*j
                number = make_data_and_label(osr, part_x, part_y, part_w, part_h, number, downsamples, part_dstime, mask, xml_bbox, part_size*(2**part_dstime))
    #just for watching contours bbox------
    for i in range(begin_level_dstime):
        img = cv2.pyrDown(img)
    cv2.imwrite(inputInfoDir+filename+'_'+'img_level'+str(LEVEL3)+'_ds'+str(begin_level_dstime)+'_bbox.jpg', img)
    #-------------------------------------

    return number


if __name__ == "__main__":
    number = 0
    begin_level_dstime = 3
    part_size = 512
    part_step = 256
    part_dstime = 3
    
    for i in range(110):
        index = i+1
        c = str(index)
        ze = 3-len(c)
        filename = 'Tumor_'+'0'*ze+c
        print filename
        osr = OpenSlide(originTrainDir+filename+'.tif')
        mask = OpenSlide(originMaskDir+filename+'_Mask.tif')
        downsamples = []
        x0, y0 =osr.level_dimensions[0]
        for k in range(osr.level_count):
            x, y = osr.level_dimensions[k]
            ds_x = float(x0*1.0/x)
            ds_y = float(y0*1.0/y)
            downsamples.append([ds_x, ds_y])
            #print k, osr.level_dimensions[k], [ds_x, ds_y]
        xml_bbox = makevocxml.parsexml(originXmlDir+filename+'.xml')
        number = just_do_it(osr, number, begin_level_dstime, downsamples, part_size, part_step,  part_dstime, mask, xml_bbox)
        osr.close()
        mask.close()
        print number
    
    writetxt.writetxt(number, 0.8, 0.5, inputMainDir)

