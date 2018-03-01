#!/usr/bin/env python
# coding=utf-8
'''
将dcgan的生成图像拆分并降采样至level-3

'''

import os
import cv2
import tools

inputdir = '../../data/dcgan/origin/'
outputdir = '../../data/dcgan/dcgan_micro_512/'
tools.mkdir(outputdir)

def preproccess(img_name, dstime, number):
    image = cv2.imread(img_name)
    print image.shape
    for i in range(2):
        for j in range(3):
            x1 = i*512
            y1 = j*512
            x2 = x1 + 512
            y2 = y1 + 512
            #img = image[0:0, 512:512]
            img = image[x1:x2, y1:y2]
            print img.shape
            for k in range(dstime):
                img = cv2.pyrDown(img)
            number += 1
            cv2.imwrite(outputdir+str(number)+'.jpg', img)
    return number


if __name__ == "__main__":
    number = 0
    dstime = 3
    for parent, dirnames, filenames in os.walk(inputdir):
        for f in filenames:
            print f
            number = preproccess(inputdir+f, dstime, number)
