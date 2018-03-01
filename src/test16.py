#!/usr/bin/env python
# coding=utf-8


import openslide
from openslide import OpenSlide
from PIL import Image
import cv2
import numpy as np
import sys
import time
import detection

#testdataDir = '../../mydata/camelyon16/Testset/'
#testimageDir = testdataDir+'Images/'
#groundtruthDir = testdataDir+'Ground_Truth/Masks/'
#infoDir = '../testdata/info/'
#outputDir = ''

singletest = '../../mydata/singletest/'
testdataDir = ''+singletest
testimageDir = ''+singletest
groundtruthDir = ''+singletest
infoDir = ''+singletest
outputDir = ''+singletest

filename = ''
PATH_TO_CKPT = '../../mydata/tfmodels/ssd_inception_v2/train/output_inference_graph.pb'
PATH_TO_LABELS = '../../mydata/tfdata/pascal_label_map.pbtxt'
NUM_CLASSES = 1

def region_newlevel(x,y,w,h,l1,l2,ds1,ds2):
    rate_x = ds1[0] / ds2[0]
    rate_y = ds1[1] / ds2[1]
    xx = int(x*ds1[0])
    yy = int(y*ds1[1])
    ww = int(w*rate_x)
    hh = int(h*rate_y)
    return xx,yy,ww,hh

def get_contours_by_th_from_beginlevel(osr, begin_level, dstime = 3, kernel_para = 15):
    img = osr.read_region((0,0), begin_level,  osr.level_dimensions[begin_level])
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
    cv2.imwrite(infoDir+filename+'_'+'contours_level'+str(begin_level)+'_ds'+str(dstime)+'.jpg', th)
    for i in range(dstime):
        th = cv2.pyrUp(th)
        img = cv2.pyrUp(img)
    image, contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, img


def part_detect(osr, x, y, w, h, first_level, downsamples, part_dstime, mask, score_thresh, num):
    part_mask = mask.read_region((x,y), first_level, (w,h))
    part_mask = cv2.cvtColor(np.array(part_mask), cv2.COLOR_RGB2GRAY)
    part = osr.read_region((x,y), first_level, (w,h))
    part = cv2.cvtColor(np.array(part), cv2.COLOR_RGB2BGR)
    for i in range(part_dstime):
        part = cv2.pyrDown(part)
        part_mask = cv2.pyrDown(part_mask)
    part = Image.fromarray(cv2.cvtColor(part, cv2.COLOR_BGR2RGB))
    part_vis, boxes, scores = detection.region_detection(PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES, part, score_thresh)
    cv2.imwrite(outputDir+str(num)+'test.jpg', part_vis)


def just_do_it(osr, number, begin_level, begin_level_dstime, first_level, downsamples, part_num_sqrt, part_size, part_dstime, mask, score_thresh):
    start = time.clock()
    contours, img = get_contours_by_th_from_beginlevel(osr, begin_level, begin_level_dstime)
    end = time.clock()
    print "get_contours:%f s" % (end - start)
    contours_bbox = []
    contours_bbox_unique = []
    num = 0
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
        xx,yy,ww,hh = region_newlevel(x,y,w,h,begin_level,first_level, downsamples[begin_level], downsamples[first_level])
        #print xx,yy,ww,hh
        #attention: cut the region into part_num_sqrt*part_num_sqrt parts and read them individually
        part_num_sqrt = (ww if ww>hh else hh) / part_size + 1
        part_w = ww/part_num_sqrt
        part_h = hh/part_num_sqrt
        for j in range(part_num_sqrt):
            for k in range(part_num_sqrt):
                part_x = xx + part_w*j
                part_y = yy + part_h*k
                part_detect(osr, part_x, part_y, part_w, part_h, first_level, downsamples, part_dstime, mask, score_thresh, num)
                num += 1

    #just for watching contours bbox------
    for i in range(begin_level_dstime):
        img = cv2.pyrDown(img)
    cv2.imwrite(infoDir+filename+'_'+'img_level'+str(begin_level)+'_ds'+str(begin_level_dstime)+'_bbox.jpg', img)
    #-------------------------------------

    return


if __name__ == "__main__":
    number = 0
    begin_level = 3
    begin_level_dstime = 3
    first_level = 0
    part_num_sqrt = 4
    part_size = 4000
    part_dstime = 3
    score_thresh = 0.5
    
    start_total = time.clock()
    for i in range(57,58):
        index = i+1
        c = str(index)
        ze = 3-len(c)
        filename = 'Test_'+'0'*ze+c
        #print filename
        osr = OpenSlide(testdataDir+filename+'.tif')
        mask = OpenSlide(groundtruthDir+filename+'_Mask.tif')
        downsamples = []
        x0, y0 =osr.level_dimensions[0]
        for k in range(osr.level_count):
            x, y = osr.level_dimensions[k]
            ds_x = float(x0*1.0/x)
            ds_y = float(y0*1.0/y)
            downsamples.append([ds_x, ds_y])
            #print k, osr.level_dimensions[k], [ds_x, ds_y]
        
        just_do_it(osr, number, begin_level, begin_level_dstime, first_level, downsamples, part_num_sqrt, part_size, part_dstime, mask, score_thresh)
        osr.close()
        mask.close()
    end_total = time.clock()
    print 'total time:%f s' %(end_total - start_total)

