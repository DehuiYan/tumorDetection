#!/usr/bin/env python
# coding=utf-8

import os
import sys
import time


def mkdir(path):
    path = path.strip()
    path = path.rstrip("/")
    if not os.path.exists(path):
        os.makedirs(path)
        print 'make path   '+path+'   success'
        return True
    else:
        print path+'   already existed'
        return False


if __name__ == "__main__":
    path = 'pathtest'
    path += '_'+time.ctime() + '/'
    mkdir(path)
