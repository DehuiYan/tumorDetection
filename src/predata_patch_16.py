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

originDir = '../../mydata/camelyon16/TrainingData/'
inputDir = '../../mydata/segdata_origin/'
originMaskDir = originDir+'Ground_Truth/Mask/'
originTrainDir = originDir+'Train_Tumor/'
originXmlDir = originDir+'Ground_Truth/XML/'
inputPatchDir = inputDir+'patch/'
inputInfoDir = inputDir+'info/'
inputMaskDir = inputDir+'mask/'
filename = ''

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
    cv2.imwrite(inputInfoDir+filename+'_'+'contours_level'+str(begin_level)+'_ds'+str(dstime)+'.jpg', th)
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


def make_patch_and_mask(osr, x, y, w, h, number, first_level, mask):
    patch = osr.read_region((x,y), first_level, (w,h))
    patch = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2BGR)
    tumor_mask = mask.read_region((x,y), first_level, (w,h))
    tumor_mask = cv2.cvtColor(np.array(tumor_mask), cv2.COLOR_RGB2GRAY)
    
    mask = np.zeros((w,h,3), np.uint8)
    white_area = 0.0
    white_th = 220
    for i in range(h):
        for j in range(w):
            if patch[i,j,0] > white_th and patch[i,j,1] > white_th and patch[i,j,2] > white_th:
                mask[i,j] = [0,0,0]
                white_area += 1.0
            elif tumor_mask[i,j] == 255:
                mask[i,j] = [255,255,255]
            else:
                mask[i,j] = [150,0,150]
    if white_area/float(w*h) > 0.5:
        return number

    number += 1
    c = str(number)
    ze = 6 - len(c)
    cv2.imwrite(inputPatchDir+'0'*ze + c +'.png', patch)
    cv2.imwrite(inputMaskDir+'0'*ze + c +'.png', mask)
    return number


def just_do_it(osr, number, begin_level, begin_level_dstime, first_level, downsamples, mask, step, xml_bbox):
    for i in xml_bbox:
        xx = i[0]
        yy = i[1]
        ww = i[2]
        hh = i[3]
        for ii in range(hh/step+1):
            for jj in range(ww/step+1):
                patch_x = xx + jj*step
                patch_y = yy + ii*step
                patch_w = step
                patch_h = step
                number = make_patch_and_mask(osr, patch_x, patch_y, patch_w, patch_h, number, first_level, mask)
    return number
    '''
    start = time.clock()
    contours, img = get_contours_by_th_from_beginlevel(osr, begin_level, begin_level_dstime)
    end = time.clock()
    print "get_contours:%f s" % (end - start)
    contours_bbox = []
    for i, cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        contours_bbox.append([x,y,w,h])

    
    for i in contours_bbox:
        x = i[0]
        y = i[1]
        w = i[2]
        h = i[3]
        #print x, y, w, h
        xx,yy,ww,hh = region_newlevel(x,y,w,h,begin_level,first_level, downsamples[begin_level], downsamples[first_level])
        #print xx,yy,ww,hh
        flag = False
        for j in xml_bbox:
            if intersection([xx,yy,ww,hh], [j[0],j[1],j[2],j[3]]) != 0:
                flag = True
                break
        if flag == False:
            continue

        for ii in range(hh/step+1):
            for jj in range(ww/step+1):
                patch_x = xx + ii*step
                patch_y = yy + jj*step
                patch_w = step
                patch_h = step
                number = make_patch_and_mask(osr, patch_x, patch_y, patch_w, patch_h, number, first_level, mask)
    return number
    '''


if __name__ == "__main__":
    number = 0
    begin_level = 3
    begin_level_dstime = 3
    first_level = 0
    step = 500
    
    start_total = time.clock()
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
        number = just_do_it(osr, number, begin_level, begin_level_dstime, first_level, downsamples, mask, step, xml_bbox)
        osr.close()
        mask.close()
        print number
    end_total = time.clock()
    print 'total time:%f s' %(end_total - start_total)



