#!/usr/bin/env python
# coding=utf-8


import sys
sys.path.append('/opt/ASAP/bin')
import openslide
from openslide import OpenSlide
import multiresolutionimageinterface as mir
from PIL import Image
import cv2
import numpy as np
import os
import time
import makevocxml
import writetxt


originDir = '../../mydata/camelyon17/'
inputDir = '../../mydata/VOCdevkit/VOC17/'
originMaskDir = originDir+'data_part/mask/'
originTrainDir = originDir+'tif/'
originXmlDir = originDir+'data_part/xml/'
inputPartDir = inputDir+'JPEGImages/'
inputPartBboxDir = inputDir+'JPEGImages_bbox/'
inputXmlDir = inputDir+'Annotations/'
inputInfoDir = inputDir+'info/'
inputMaskDir = inputDir+'Mask/'
filename = ''

LEVEL0 = 0
LEVEL3 = 3
LEVEL6 = 6

def region_translevel(x,y,w,h,l1,l2):
    zoom = 2**l1 / 2**l2
    xx = int(x*zoom)
    yy = int(y*zoom)
    ww = int(w*zoom)
    hh = int(h*zoom)
    return xx,yy,ww,hh

def get_tissue(osr, mr_image):
    dstime = LEVEL6 - LEVEL3
    kernel_para = 15

    if osr.level_count == 1:
        level = LEVEL6
        size = (osr.level_dimensions[0][0]/(2**LEVEL6), osr.level_dimensions[0][1]/(2**LEVEL6))
    elif osr.level_count == 10 or (osr.level_count == 9 and osr.level_dimensions[8][1] == 512):
        level = LEVEL3
        size = osr.level_dimensions[LEVEL3]
    else:
        level = LEVEL6
        size = osr.level_dimensions[LEVEL6]

    #img = osr.read_region((0,0), level, size)
    ds = mr_image.getLevelDownsample(level)
    img = mr_image.getUCharPatch(0, 0, size[0], size[1], level)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if osr.level_count == 10 or (osr.level_count == 9 and osr.level_dimensions[8][1] == 512):
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
    cv2.imwrite(inputInfoDir+filename+'_tissue.jpg', th)

    image, contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tissue = []
    for i, cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h < 10000:
            continue
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),8)
        xx,yy,ww,hh = region_translevel(x,y,w,h,LEVEL6,LEVEL3)
        print x,y,w,h
        print xx,yy,ww,hh
        tissue.append([xx,yy,ww,hh])
    cv2.imwrite(inputInfoDir+filename+'_tissue_bbox.jpg', img)
    
    return tissue


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

def make_data_and_label(osr, x, y, w, h, number, part_dstime, mask, xml_bbox):
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

    number += 1
    bbox_label_this = []
    part = osr.read_region((x,y), LEVEL0, (w,h))
    part = cv2.cvtColor(np.array(part), cv2.COLOR_RGB2BGR)
    part_bbox = part.copy()
    for i, cnt in enumerate(contours):
        mask_x, mask_y, mask_w, mask_h = cv2.boundingRect(cnt)
        #print mask_x, mask_y, mask_w, mask_h
        area = cv2.contourArea(cnt)
        extent = float(area)/(mask_w*mask_h)
        bbox_label_this.append([mask_x, mask_y, mask_x+mask_w, mask_y+mask_h, area, extent])
        part_bbox = cv2.rectangle(part_bbox, (mask_x, mask_y), (mask_x+mask_w, mask_y+mask_h), (0,255,0), 32)

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


def just_do_it(osr, mr_image, number, part_size, part_dstime, mask, xml_bbox):
    start = time.clock()
    tissue = get_tissue(osr, mr_image)
    end = time.clock()
    print "get_tissue:%f s" % (end - start)
    num = 0
    for i in tissue:
        x = i[0]
        y = i[1]
        w = i[2]
        h = i[3]
        xx, yy, ww, hh = region_translevel(x,y,w,h,LEVEL3,LEVEL0)
        part_num_sqrt = (ww if ww>hh else hh) / part_size + 1
        part_w = ww/part_num_sqrt
        part_h = hh/part_num_sqrt
        for j in range(part_num_sqrt):
            for k in range(part_num_sqrt):
                part_x = xx + part_w*j
                part_y = yy + part_h*k
                number = make_data_and_label(osr, part_x, part_y, part_w, part_h, number, part_dstime, mask, xml_bbox)
    return number


if __name__ == "__main__":
    number = 0
    part_size = 4000
    part_dstime = 3
    
    start_total = time.clock()

    for parent, dirnames, filenames in os.walk(originXmlDir):
        for f in filenames:
            filename = f.split('.')[0]
            #print filename
            osr = OpenSlide(originTrainDir+filename+'.tif')
            mask = OpenSlide(originMaskDir+filename+'.mask')
            reader = mir.MultiResolutionImageReader()
            mr_image = reader.open(originTrainDir+filename+'.tif')
            xml_bbox = makevocxml.parsexml(originXmlDir+filename+'.xml')
            for k in range(osr.level_count):
                print k, osr.level_dimensions[k], osr.level_downsamples[k]
            
            number = just_do_it(osr, mr_image, number, part_size, part_dstime, mask, xml_bbox)
            osr.close()
            mask.close()
    end_total = time.clock()
    print 'total time:%f s' %(end_total - start_total)

