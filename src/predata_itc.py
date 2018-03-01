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
import os
import math

inputDir1 = '../../mydata/camelyon17/'
inputTrainDir1 = inputDir1+'tif/'
inputXmlDir1 = inputDir1+'data_part/xml/'
inputDir2 = '../../mydata/camelyon16/'
inputTrainDir2 = inputDir2+'TrainingData/Train_Tumor/'
inputXmlDir2 = inputDir2+'TrainingData/Ground_Truth/XML/'
inputDir3 = '../../mydata/camelyon16/'
inputTrainDir3 = inputDir3+'Testset/Images/'
inputXmlDir3 = inputDir3+'Testset/Ground_Truth/Annotations/'

outputDir = '../../mydata/itc/'
outputItcDir = outputDir+'origin/'
filename = ''

def make_patch_and_mask(osr, x, y, w, h, number):
    patch = osr.read_region((x,y), 0, (w,h))
    patch = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2BGR)
    number += 1
    c = str(number)
    ze = 6 - len(c)
    cv2.imwrite(outputItcDir+'0'*ze + c +'.png', patch)
    return number


def just_do_it(osr, number, xml_bbox):
    for i in xml_bbox:
        xx = i[0]
        yy = i[1]
        ww = i[2]
        hh = i[3]
        #print xx, yy, ww, hh
        '''
        patch_x = xx - ww/2
        if patch_x < 0:
            patch_x = 0
        patch_y = yy - hh/2
        if patch_y < 0:
            patch_y = 0
        patch_w = ww * 2
        if patch_x + patch_w > osr.dimensions[0]:
            patch_w = osr.dimensions[0] - patch_x
        patch_h = hh * 2
        if patch_y + patch_h > osr.dimensions[1]:
            patch_h = osr.dimensions[1] - patch_y
        #print patch_x, patch_y, patch_w, patch_h
        '''
        if math.sqrt(ww**2 + hh**2) < 850:
            patch_x = xx
            patch_y = yy
            patch_w = ww
            patch_h = hh
            number = make_patch_and_mask(osr, patch_x, patch_y, patch_w, patch_h, number)
    return number


if __name__ == "__main__":
    number = 0
    
    for parent, dirnames, filenames in os.walk(inputXmlDir1):
        for f in filenames:
            filename = f.split('.')[0]
            osr = OpenSlide(inputTrainDir1+filename+'.tif')
            xml_bbox = makevocxml.parsexml(inputXmlDir1+filename+'.xml')
            print filename
            number = just_do_it(osr, number, xml_bbox)
            osr.close()
    for parent, dirnames, filenames in os.walk(inputXmlDir2):
        for f in filenames:
            filename = f.split('.')[0]
            osr = OpenSlide(inputTrainDir2+filename+'.tif')
            xml_bbox = makevocxml.parsexml(inputXmlDir2+filename+'.xml')
            print filename
            number = just_do_it(osr, number, xml_bbox)
            osr.close()
    for parent, dirnames, filenames in os.walk(inputXmlDir3):
        for f in filenames:
            filename = f.split('.')[0]
            osr = OpenSlide(inputTrainDir3+filename+'.tif')
            xml_bbox = makevocxml.parsexml(inputXmlDir3+filename+'.xml')
            print filename
            number = just_do_it(osr, number, xml_bbox)
            osr.close()


