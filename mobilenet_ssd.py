
#coding=utf-8

import numpy as np
import sys, os
import cv2



import time


caffe_root = '/home/gjw/caffe-ssd-mobile/'
sys.path.insert(0, caffe_root + 'python')

import caffe
caffe.set_mode_gpu()  ### 设置GPU模式


CLASSES = ('background',
           'car', 'cyclist', 'pedestrain')


# 全局变量
colours = np.random.rand(32,3)*255


class MobileNet_SSD:
    # 构造函数
    def __init__(self, net_file,caffe_model):
        self.net = caffe.Net(net_file, caffe_model, caffe.TEST)

    # 图像归一化
    def preprocess(self, src):
        img = cv2.resize(src, (300, 300))
        return (img - 127.5) * 0.007843

    def detect(self, frame):
        ###  ssd检测开始  ###
        # 1. 前向传播之前的图像预处理
        img = self.preprocess(frame)

        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))

        self.net.blobs['data'].data[...] = img

        # 2. 前向传播，取'detection_out'层，结果存放到detections中
        detections = self.net.forward()['detection_out']

        height = frame.shape[0]
        width = frame.shape[1]

        # 3.从detections中，获取相关信息
        cls = detections[0, 0, :, 1]  # 类别
        conf = detections[0, 0, :, 2]  # 置信度
        box = (detections[0, 0, :, 3:7] * np.array([width, height, width, height])).astype(np.int32) #目标框Rect

        result = []
        for i in range(len(box)):
            if(conf[i] > 0.2):  # 置信度阈值
                LeftTop = (box[i][0], box[i][1])  # 左上角坐标
                RightBottom = (box[i][2], box[i][3])  # 右下角坐标
                result.append([box[i][0], box[i][1], box[i][2], box[i][3], conf[i]])  # 左上角，右下角，置信度
        result = np.array(result)  # 提出result，给det

        det = []
        if result != []:  # 非常重要，非空判断
            det = result[:, 0:5]
        return det

    ###  ssd检测结束  ###


