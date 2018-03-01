#!/usr/bin/env python
# coding=utf-8
import sys
sys.path.append('/opt/ASAP/bin')

import multiresolutionimageinterface as mir
import openslide
from openslide import OpenSlide
from PIL import Image
import cv2
import numpy as np
import time
import detection
import os

testdataDir = '../../mydata/camelyon17/'
testtifDir = testdataDir+'tif/'
xmlDir = testdataDir+'data_part/xml/'
maskDir = testdataDir+'data_part/mask/'
infoDir = testdataDir+'data_part/info/'
outputDir = testdataDir+'data_part/output0220/'

filename = ''
PATH_TO_CKPT = '../../models/object_detection/VOC2007/output_inference_graph.pb'
PATH_TO_LABELS = '../../models/object_detection/VOC2007/pascal_label_map.pbtxt'
NUM_CLASSES = 1
#PATH_TO_CKPT = '../../mydata/tfmodels/ssd_inception_v2/train/output_inference_graph.pb'
#PATH_TO_LABELS = '../../mydata/tfdata/pascal_label_map.pbtxt'
#NUM_CLASSES = 1

LEVEL0 = 0
LEVEL3 = 3
LEVEL6 = 6
TPTNFPFN = [0, 0, 0, 0]

def region_translevel(x,y,w,h,l1,l2):
    zoom = 2**l1 / 2**l2
    xx = int(x*zoom)
    yy = int(y*zoom)
    ww = int(w*zoom)
    hh = int(h*zoom)
    return xx,yy,ww,hh

def get_tissue(osr, mr_image):
    dstime = LEVEL6 - LEVEL3
    tissue = []

    if osr.level_count == 1:
        level = LEVEL6
        size = (osr.level_dimensions[0][0]/(2**LEVEL6), osr.level_dimensions[0][1]/(2**LEVEL6))
    elif osr.level_count == 10 or (osr.level_count == 9 and osr.level_dimensions[8][1] == 512):
        ##return tissue
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
    kernel_para = 15
    kernel = np.ones((kernel_para, kernel_para),np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    #show threshold segment result
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(infoDir+filename+'_tissue.jpg', th)

    image, contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h < 10000:
            continue
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),8)
        xx,yy,ww,hh = region_translevel(x,y,w,h,LEVEL6,LEVEL3)
        #print w,h
        #print ww,hh
        #print ' '
        tissue.append([xx,yy,ww,hh])
    cv2.imwrite(infoDir+filename+'_tissue_bbox.jpg', img)
    
    return tissue


def part_detect(osr, mr_image, x, y, w, h, level, score_thresh, number):
    ds = mr_image.getLevelDownsample(level)
    part = mr_image.getUCharPatch(int(x * ds), int(y * ds), w, h, level)
    part = cv2.cvtColor(np.array(part), cv2.COLOR_RGB2BGR)

    part_groundtruth = part.copy()
    mr_mask = OpenSlide(maskDir+filename+'.mask')
    part_mask = mr_mask.read_region((int(x * ds), int(y * ds)), level, (w,h))
    part_mask = cv2.cvtColor(np.array(part_mask), cv2.COLOR_RGB2GRAY)
    #for i in range(part_dstime):
    #    part = cv2.pyrDown(part)
    #    part_mask = cv2.pyrDown(part_mask)
    part = Image.fromarray(cv2.cvtColor(part, cv2.COLOR_BGR2RGB))
    start = time.clock()
    flag1, part_vis, boxes, scores = detection.region_detection(PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES, part, score_thresh)
    end = time.clock()
    print "detection:%f s" % (end - start)
    part_groundtruth = cv2.cvtColor(part_groundtruth, cv2.COLOR_BGR2RGB)
    image, contours, hierarchy = cv2.findContours(part_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        flag2 = True
    else:
        flag2 = False
    for i, cnt in enumerate(contours):
        mask_x, mask_y, mask_w, mask_h = cv2.boundingRect(cnt)
        part_groundtruth = cv2.rectangle(part_groundtruth, (mask_x, mask_y), (mask_x+mask_w, mask_y+mask_h), (255,0,0), 8)
    if flag1 == True or flag2 == True:
        cv2.imwrite(outputDir+filename+'_'+str(number)+'test.jpg', part_vis)
        cv2.imwrite(outputDir+filename+'_'+str(number)+'test_groundtruth.jpg', part_groundtruth)
        cv2.imwrite(outputDir+filename+'_'+str(number)+'test_mask.jpg', part_mask)
    if flag1 == True and flag2 == True:
        TPTNFPFN[0] += 1
    elif flag1 == False and flag2 == False:
        TPTNFPFN[1] += 1
    elif flag1 == True and flag2 == False:
        TPTNFPFN[2] += 1
    elif flag1 == False and flag2 == True:
        TPTNFPFN[3] += 1
    print TPTNFPFN


def just_do_it(osr, mr_image, part_size, score_thresh):
    start = time.clock()
    tissue = get_tissue(osr, mr_image)
    end = time.clock()
    print "get_tissue:%f s" % (end - start)
    number = 0
    for i in tissue:
        x = i[0]
        y = i[1]
        w = i[2]
        h = i[3]
        part_w = part_size
        part_h = part_size
        for j in range(w/part_size + 1):
            for k in range(h/part_size + 1):
                part_x = x + part_w*j
                part_y = y + part_h*k
                part_detect(osr, mr_image, part_x, part_y, part_w, part_h, LEVEL3, score_thresh, number)
                number += 1


if __name__ == "__main__":
    part_size = 512
    score_thresh = 0.3
    for parent, dirnames, filenames in os.walk(xmlDir):
        for f in filenames:
            filename = f.split('.')[0]
            osr = OpenSlide(testtifDir+filename+'.tif')
            reader = mir.MultiResolutionImageReader()
            mr_image = reader.open(testtifDir+filename+'.tif')
            #for k in range(osr.level_count):
                #print k, osr.level_dimensions[k], osr.level_downsamples[k]
            print filename
            #print osr.level_count
            start_total = time.clock()
            just_do_it(osr, mr_image, part_size, score_thresh)
            end_total = time.clock()
            print 'total time:%f s' %(end_total - start_total)
            osr.close()
    print TPTNFPFN
    TP = TPTNFPFN[0]
    TN = TPTNFPFN[1]
    FP = TPTNFPFN[2]
    FN = TPTNFPFN[3]
    print 'precision:'+str(TP/(TP+FP))
    print 'recal:'+str(TP/(TP+FN))
