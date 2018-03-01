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
import tools
import math

inputDir = '../../mydata/camelyon17/'
inputMaskDir = inputDir+'mask/'
inputTifDir = inputDir+'tif/'
inputXmlDir = inputDir+'xml/'

outputDir = '../../mydata/camelyon17/tumor/'
outputItcDir = outputDir+'itc_tumor/'
outputItcMaskDir = outputDir+'itc_mask/'
outputMicroDir = outputDir+'micro_tumor/'
outputMicroMaskDir = outputDir+'micro_mask/'
outputMacroDir = outputDir+'macro_tumor/'
outputMacroMaskDir = outputDir+'macro_mask/'
tools.mkdir(outputDir)
tools.mkdir(outputItcDir)
tools.mkdir(outputItcMaskDir)
tools.mkdir(outputMicroDir)
tools.mkdir(outputMicroMaskDir)
tools.mkdir(outputMacroDir)
tools.mkdir(outputMacroMaskDir)
filename = ''


def make_tumor_and_mask(osr, x, y, w, h, number, mask, outputTumorDir, outputMaskDir):
    tumor = osr.read_region((x,y), 0, (w,h))
    tumor = cv2.cvtColor(np.array(tumor), cv2.COLOR_RGB2BGR)
    tumor_mask = mask.read_region((x,y), 0, (w,h))
    tumor_mask = cv2.cvtColor(np.array(tumor_mask), cv2.COLOR_RGB2GRAY)
    '''
    mask = np.zeros((h,w,3), np.uint8)
    white_th = 220
    for i in range(h):
        for j in range(w):
            if tumor[i,j,0] > white_th and tumor[i,j,1] > white_th and tumor[i,j,2] > white_th:
                mask[i,j] = [0,0,0]
            elif tumor_mask[i,j] == 255:
                mask[i,j] = [255,255,255]
            else:
                mask[i,j] = [150,0,150]
    '''
    number += 1
    c = str(number)
    ze = 6 - len(c)
    cv2.imwrite(outputTumorDir+'0'*ze + c +'.png', tumor)
    cv2.imwrite(outputMaskDir+'0'*ze + c +'.png', tumor_mask)
    return number


def just_do_it(osr, number, mask, xml_bbox):
    for i in xml_bbox:
        x = i[0]
        y = i[1]
        w = i[2]
        h = i[3]
        if math.sqrt(w**2+h**2) < 850:
            outputTumorDir = outputItcDir
            outputMaskDir = outputItcMaskDir
        elif math.sqrt(w**2+h**2) > 8500:
            outputTumorDir = outputMacroDir
            outputMaskDir = outputMacroMaskDir
        else:
            outputTumorDir = outputMicroDir
            outputMaskDir = outputMicroMaskDir
        number = make_tumor_and_mask(osr, x, y, w, h, number, mask, outputTumorDir, outputMaskDir)
    return number


if __name__ == "__main__":
    number = 0

    for parent, dirnames, filenames in os.walk(inputXmlDir):
        for f in filenames:
            filename = f.split('.')[0]
            osr = OpenSlide(inputTifDir+filename+'.tif')
            mask = OpenSlide(inputMaskDir+filename+'.mask')
            xml_bbox = makevocxml.parsexml(inputXmlDir+filename+'.xml')
            print filename
            number = just_do_it(osr, number, mask, xml_bbox)
            osr.close()
            mask.close()
    print number



