#!/usr/bin/env python
# coding=utf-8


import openslide
from openslide import OpenSlide
from PIL import Image
import cv2
import numpy as np
import sys
import tools
import math
import random
import os


inputDir = '../../mydata/camelyon16/TrainingData/Train_Normal/'
outputPartDir = '../../mydata/normal/part/'
outputPatchDir = '../../mydata/patch/normal/'
outputInfoDir = '../../mydata/normal/info/'

tools.mkdir(inputDir)
tools.mkdir(outputPartDir)
tools.mkdir(outputPatchDir)
tools.mkdir(outputInfoDir)

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

    for i in range(dstime):
        th = cv2.pyrUp(th)

    image, contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours 


def make_part(osr, x, y, w, h, number1):
    xx,yy,ww,hh = region_newlevel(x,y,w,h,LEVEL3,LEVEL0, downsamples[LEVEL3], downsamples[LEVEL0])
    part = osr.read_region((xx,yy), LEVEL3, (w,h))
    part = cv2.cvtColor(np.array(part), cv2.COLOR_RGB2BGR)
    number1 += 1
    c = str(number1)
    ze = 6 - len(c)
    cv2.imwrite(outputPartDir+'0'*ze + c +'.jpg', part)
    return number1


def make_patch(osr, x, y, w, h, number2):
    patch = osr.read_region((x,y), LEVEL0, (w,h))
    patch = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2BGR)
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(patch_hsv)
    ret, binary = cv2.threshold(s, 30, 255, cv2.THRESH_BINARY)
    nonzero = cv2.countNonZero(binary)
    if nonzero < 200 * 200:
        return number2
    
    r = random.randint(1,30)
    if r < 30:
        return number2
    
    number2 += 1
    c = str(number2)
    ze = 6 - len(c)
    cv2.imwrite(outputPatchDir+'0'*ze+c+'.jpg', patch)
    return number2


def just_do_it(osr, number1, number2, begin_level_dstime, downsamples, part_size, part_step, part_dstime, patch_size):
    contours = get_tissue(osr, begin_level_dstime)
    contours_bbox = []
    for i, cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        contours_bbox.append([x,y,w,h])
    for i in contours_bbox:
        x = i[0]
        y = i[1]
        w = i[2]
        h = i[3]
        '''
        part_w = part_size
        part_h = part_size
        xx = x + (w%part_step)/2
        yy = y + (h%part_step)/2
        for j in range(h/part_step):
            for k in range(w/part_step):
                part_x = xx + part_step*k
                part_y = yy + part_step*j
                number1 = make_part(osr, part_x, part_y, part_w, part_h, number1)
        '''
        xx,yy,ww,hh = region_newlevel(x,y,w,h,LEVEL3,LEVEL0, downsamples[LEVEL3], downsamples[LEVEL0])
        patch_w = patch_size
        patch_h = patch_size
        xx = xx + (ww%patch_size)/2
        yy = yy + (yy%patch_size)/2
        for j in range(hh/patch_size):
            for k in range(ww/patch_size):
                patch_x = xx + patch_size*k
                patch_y = yy + patch_size*j
                number2 = make_patch(osr, patch_x, patch_y, patch_w, patch_h, number2)
    return number1, number2


if __name__ == "__main__":
    number1 = 0
    number2 = 0
    begin_level_dstime = 3
    part_size = 512
    part_step = 512
    patch_size = 256
    part_dstime = 3
    
    for parent, dirnames, filenames in os.walk(inputDir):
        for f in filenames:
            filename = f.split('.')[0]
            print filename
            osr = OpenSlide(inputDir+filename+'.tif')
            downsamples = []
            x0, y0 =osr.level_dimensions[0]
            for k in range(osr.level_count):
                x, y = osr.level_dimensions[k]
                ds_x = float(x0*1.0/x)
                ds_y = float(y0*1.0/y)
                downsamples.append([ds_x, ds_y])
                #print k, osr.level_dimensions[k], [ds_x, ds_y]
            
            number1, number2 = just_do_it(osr, number1, number2, begin_level_dstime, downsamples, part_size, part_step,  part_dstime, patch_size)
            osr.close()




