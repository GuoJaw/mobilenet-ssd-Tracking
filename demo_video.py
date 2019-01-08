#coding=utf-8
import os
import sys
import cv2
import numpy as np
import argparse
import time  # fps


caffe_root = '/home/gjw/caffe-ssd-mobile/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  



from mobilenet_ssd import MobileNet_SSD
from sort import Sort

def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()

    # mobilenet-ssd的参数
    parser.add_argument('--net_file',default='./model/MobileNetSSD_deploy.prototxt')
    parser.add_argument('--caffe_model',default='./model/MobileNetSSD_deploy.caffemodel')
    
    #parser.add_argument('--net_file',default='./model_kitti/MobileNetSSD_deploy.prototxt')
    #parser.add_argument('--caffe_model',default='./model_kitti/MobileNetSSD_final.caffemodel')
    # track_sort的参数
    parser.add_argument('--sort_max_age',default=5,type=int)
    parser.add_argument('--sort_min_hit',default=3,type=int)

    return parser.parse_args()



# 全局变量
colours = np.random.rand(32,3)*255

if __name__=="__main__":
    args=parse_args()

    ## track_sort初始化
    mot_tracker = Sort(args.sort_max_age, args.sort_min_hit)

    # 检测器： 创建MobileNet_SSD类型的变量SSD
    SSD_Object = MobileNet_SSD(args.net_file, args.caffe_model)  # 加载caffe网络模型,初始化
    cap = cv2.VideoCapture('./MOT06.mp4')  # 读取视频

    while (1) :
        ret, frame = cap.read()
        if ret is False:
            print("load video or capture error !")
            break

        start = time.time()  # fps开始时间

####
        det = SSD_Object.detect(frame) # ssd检测
####

####
        trackers = mot_tracker.update(det) # 用mot_tracker的update接口去更新det，进行多目标的跟踪

        for track in trackers:
            # 左上角坐标(x,y)
            lrx=int(track[0])
            lry=int(track[1])

            # 右下角坐标(x,y)
            rtx=int(track[2])
            rty=int(track[3])

            #track_id
            trackID=int(track[4])

            cv2.putText(frame, str(trackID), (lrx,lry), cv2.FONT_ITALIC, 0.6, (int(colours[trackID%32,0]),int(colours[trackID%32,1]),int(colours[trackID%32,2])),2)
            cv2.rectangle(frame,(lrx,lry),(rtx,rty),(int(colours[trackID%32,0]),int(colours[trackID%32,1]),int(colours[trackID%32,2])),2)
####

        end = time.time() # fps结束时间
        fps = 1 / (end - start);
        print('FPS = %.2f' %(fps))


        #显示图像
        #frame = cv2.resize(frame, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # 图像放大为原来两倍
        cv2.imshow("frame",frame)
        if cv2.waitKey(1)& 0xFF == ord('q'):
            break



