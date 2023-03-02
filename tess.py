# coding=utf-8
import os

import cv2
import numpy as np

import mvsdk
import platform
import time

dirs = ['./l1.5','./r1.5']
n = 0

camera_matrix1 = np.array([[650.346616444449, 0, 0],
[0, 649.472545560264, 0],
[659.364563511604, 351.966016907267, 1]]).T
camera_matrix2 = np.array([[653.100835819782, 0, 0],
[0, 652.374708058560, 0],
[662.890788905584, 363.935569940028, 1]]).T
# (k1, k2, p1, p2, k3)
distCoeff1 = np.array([0.185736431470733, -0.250383574833521, 0, 0, 0.113275893049391])
distCoeff2 = np.array([0.189161563576057, -0.254669885642223, 0, 0, 0.115819997586013])
# 下方算式不用替换
u01 = camera_matrix1[0, 2]
u02 = camera_matrix2[0, 2]
dx_f = (1 / camera_matrix1[0, 0] + 1 / camera_matrix2[0, 0]) / 2
focus = (camera_matrix1[0, 0] + camera_matrix2[0, 0]) / 2
base = 60


DevList = mvsdk.CameraEnumerateDevice()
nDev = len(DevList)
for i, DevInfo in enumerate(DevList):
    print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
class Camera(object):
    def __init__(self, i):
        self.DevInfo = DevList[i]
        self.n = 0
        print(DevInfo)

    def open(self,a):
        # 打开相机
        hCamera = 0
        try:
            hCamera = mvsdk.CameraInit(self.DevInfo, -1, -1)
            self.hCamera = hCamera
        except mvsdk.CameraException as e:
            print("CameraInit Failed({}): {}".format(e.error_code, e.message))
            return

        # 获取相机特性描述
        cap = mvsdk.CameraGetCapability(hCamera)

        # 判断是黑白相机还是彩色相机
        monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

        # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
        if monoCamera:
            mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        # 相机模式切换成连续采集
        mvsdk.CameraSetTriggerMode(hCamera, 0)

        # 手动曝光，曝光时间30ms
        mvsdk.CameraSetAeState(hCamera, 0)
        mvsdk.CameraSetExposureTime(hCamera, 70.5307 * 1000)
        mvsdk.CameraSetAnalogGain(hCamera, 20)


        # 让SDK内部取图线程开始工作
        mvsdk.CameraPlay(self.hCamera)

        # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
        FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

        # 分配RGB buffer，用来存放ISP输出的图像
        # 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
        self.pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

    def read(self, a, num,key):
            # 从相机取帧图片q一
            # print(a)
            try:
                pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
                mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
                mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)

                # windows下取到的图像数据是上下颠倒的，以BMP格式存放。转换成opencv则需要上下翻转成正的
                # linux下直接输出正的，不需要上下翻转
                if platform.system() == "Windows":
                    mvsdk.CameraFlipFrameBuffer(self.pFrameBuffer, FrameHead, 1)

                # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
                # 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
                frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer)
                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth,
                                       1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))

                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)

                if key == 'a':
                    # 从相机取一帧图片
                    try:
                        # pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 2000)
                        # mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
                        # mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)
                        #
                        # # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
                        # # 该示例中我们只是把图片保存到硬盘文件中
                        #
                        # status = mvsdk.CameraSaveImage(self.hCamera, dirs[num] + "/pic"+str(self.n)+".bmp".format(str(self.n)), self.pFrameBuffer, FrameHead,
                        #                                mvsdk.FILE_BMP, 100)
                        # self.n += 1
                        # if status == mvsdk.CAMERA_STATUS_SUCCESS:
                        #     print("Save image successfully. image_size = {}X{}".format(FrameHead.iWidth,
                        #                                                                FrameHead.iHeight))
                        # else:
                        #     print("Save image failed. err={}".format(status))
                        print('6')
                        cv2.imwrite(dirs[num] + "/"+a+"pic"+str(self.n)+".jpg".format(str(self.n)),frame)
                        self.n += 1
                    except mvsdk.CameraException as e:
                        # print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))
                        print('no')
                return frame
            except mvsdk.CameraException as e:
                if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                    print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))
                print(e)

    def close(self):
        # 关闭相机
        mvsdk.CameraUnInit(self.hCamera)

        # 释放帧缓存
        mvsdk.CameraAlignFree(self.pFrameBuffer)

    def savePic(self, dir, state):
        if state == 's':
            # 从相机取一帧图片
            try:
                pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 2000)
                mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
                mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)

                # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
                # 该示例中我们只是把图片保存到硬盘文件中

                status = mvsdk.CameraSaveImage(self.hCamera, dir + "/grab%s.bmp".format(str(n)), self.pFrameBuffer, FrameHead, mvsdk.FILE_BMP,
                                               100)
                if status == mvsdk.CAMERA_STATUS_SUCCESS:
                    print("Save image successfully. image_size = {}X{}".format(FrameHead.iWidth, FrameHead.iHeight))
                else:
                    print("Save image failed. err={}".format(status))
            except mvsdk.CameraException as e:
                print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))



if __name__ == "__main__":
    # 枚举相机
    try:
        os.mkdir(dirs[0])
        os.mkdir(dirs[1])
    except:
        print('makesuccessfully')
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    for i, DevInfo in enumerate(DevList):
        print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
    camera_1=Camera(0)
    camera_2=Camera(1)
    camera_1.open("l")
    camera_2.open("r")
    while True:
        f1 = camera_2.read("l", 0, 'a')
        f2 = camera_1.read("r", 1, 'a')
        time.sleep(3)
    camera_2.close()
    camera_1.close()

    cv2.destroyAllWindows()
    print(1e+03)