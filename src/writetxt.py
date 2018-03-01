#!/usr/bin/env python
# coding=utf-8

import os
import random

def writetxt(total, trainval, train, path):
    #total = 3745
    items = []
    #trainval = 0.8
    #train = 0.5
    val = trainval - train
    test = 1 - trainval

    #path = '../../mydata/VOCdevkit/VOC2007/ImageSets/Main/'
    trainvaltxt = path + 'trainval.txt'
    traintxt = path + 'train.txt'
    valtxt = path + 'val.txt'
    testtxt = path + 'test.txt'

    for i in range(total):
        num = i+1
        c = str(num)
        ze = 6 - len(c)
        items.append('0'*ze + c)
    random.shuffle(items)

    trainval_items = []
    for i in range(int(total*trainval)):
        trainval_items.append(items[i])
    trainval_items.sort()
    with open(trainvaltxt, 'w') as f:
        for i in trainval_items:
            f.write(i+'\n')

    train_items = []
    for i in range(int(total*train)):
        train_items.append(items[i])
    train_items.sort()
    with open(traintxt, 'w') as f:
        for i in train_items:
            f.write(i+'\n')

    val_items = []
    for i in range(int(total*train), int(total*trainval)):
        val_items.append(items[i])
    val_items.sort()
    with open(valtxt, 'w') as f:
        for i in val_items:
            f.write(i+'\n')

    test_items = []
    for i in range(int(total*trainval), total):
        test_items.append(items[i])
    test_items.sort()
    with open(testtxt, 'w') as f:
        for i in test_items:
            f.write(i+'\n')


if __name__ == '__main__':
    total = 34924 
    trainval = 0.9
    train = 0.6
    path = '../../mydata/VOCdevkit/VOC2007/ImageSets/Main/'
    writetxt(total, trainval, train, path)

