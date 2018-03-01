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
import tools
import random

inputTrainDir = ['../../mydata/camelyon17/tif/', '../../mydata/camelyon16/TrainingData/Train_Tumor/', '../../mydata/camelyon16/Testset/Images/']
inputXmlDir = ['../../mydata/camelyon17/data_part/xml/', '../../mydata/camelyon16/TrainingData/Ground_Truth/XML/', '../../mydata/camelyon16/Testset/Ground_Truth/Annotations/']
inputMaskDir = ['../../mydata/camelyon17/data_part/mask/', '../../mydata/camelyon16/TrainingData/Ground_Truth/Mask/', '../../mydata/camelyon16/Testset/Ground_Truth/Masks/']

outputDir = '../../mydata/patch_512/'
outputItcDir = outputDir+'itc/'
outputMicroDir = outputDir+'micro/'
outputMacroDir = outputDir+'macro/'
outputMaskDir = outputDir+'mask/'
outputPatchDir = '../../mydata/patch/tumor/'
tools.mkdir(outputPatchDir)
tools.mkdir(outputDir)
tools.mkdir(outputItcDir)
tools.mkdir(outputMicroDir)
tools.mkdir(outputMacroDir)
tools.mkdir(outputMaskDir)
filename = ''

def make_patch_and_mask(osr, x, y, w, h, number, mask, outputdir, filename):
    r = random.randint(1,5)
    if r < 5:
        return number
    patch = osr.read_region((x,y), 0, (w,h))
    patch = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2BGR)
    patch_mask = mask.read_region((x,y), 0, (w,h))
    patch_mask = cv2.cvtColor(np.array(patch_mask), cv2.COLOR_RGB2GRAY)
    image, contours, hierarchy = cv2.findContours(patch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = 0
    for i, cnt in enumerate(contours):
        area += cv2.contourArea(cnt)
    extent = float(area) / (w*h)
    if extent < 0.99:
        return number
    number += 1
    c = str(number)
    ze = 6 - len(c)
    cv2.imwrite(outputdir+'0'*ze + c + filename + '.png', patch)
    #cv2.imwrite(outputMaskDir+'0'*ze + c + '.png', patch_mask)
    return number


def just_do_it(osr, number, xml_bbox, mask, step, filename, length):
    for i in xml_bbox:
        xx = i[0]
        yy = i[1]
        ww = i[2]
        hh = i[3]
        #print xx, yy, ww, h
        if ww < step or hh < step:
            return number
        '''
        elif math.sqrt(ww**2 + hh**2) < 850:
            outputdir = outputItcDir
            return number
        elif math.sqrt(ww**2 + hh**2) > 850 and math.sqrt(ww**2 + hh**2) < 8500:
            outputdir = outputMicroDir
            step = 128
        else:
            outputdir = outputMacroDir
            step = 512
        '''
        outputdir = outputPatchDir
        step = 256
        xx = xx + (ww%step)/2
        yy = yy + (hh%step)/2
        patch_w = length
        patch_h = length
        for ii in range(hh/step-length/step+1):
            for jj in range(ww/step-length/step+1):
                patch_x = xx + jj*step
                patch_y = yy + ii*step
                number = make_patch_and_mask(osr, patch_x, patch_y, patch_w, patch_h, number, mask, outputdir, filename)
    return number


if __name__ == "__main__":
    number = 0
    length = 256
    step = 256
    for i in range(3):
        if i == 0:
            continue
        for parent, dirnames, filenames in os.walk(inputXmlDir[i]):
            for f in filenames:
                filename = f.split('.')[0]
                osr = OpenSlide(inputTrainDir[i]+filename+'.tif')
                xml_bbox = makevocxml.parsexml(inputXmlDir[i]+filename+'.xml')
                if i == 0:
                    mask = OpenSlide(inputMaskDir[i]+filename+'.mask')
                else:
                    mask = OpenSlide(inputMaskDir[i]+filename+'_Mask.tif')
                print filename
                number = just_do_it(osr, number, xml_bbox, mask, step, filename, length)
                osr.close()
                mask.close()
