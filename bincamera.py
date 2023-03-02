# Libraries
import logging
import os
import cv2 as cv
import numpy as np
from typing import Union
#from camera import Camera
# Local Files
from funcins import set_fields_with_params
from returns import BinocularCameraReturn
from sensor import Sensor
from params import camera_matrix1, camera_matrix2, distCoeff1, distCoeff2



class BinocularCamera(Sensor):
    
    def __init__(self, name: str):
        super().__init__(name)
        self.source = 0
        self.width, self.height = (2560, 720)
        self.maps = None
        self.remap = True
        self.show_rectified = False
        self.__raw = (None, None) # left, Right
        self.__msg = (None, None) # left, Right
        

    #def start(self,width: int = 2560, height: int = 720, maps: Union[None, os.PathLike] = None ):
    def start(self,width: int = 2560, height: int = 720, maps: Union[None, os.PathLike] = "./maps7" ):
        """
        Params
        -----
        maps: PathLike
            标定数据文件夹
        sources: int
            打开的摄像头
        """
        
        #logging.debug("Initializing BinocularCamera({})".format(self.__name))

        # self.cap = cv.VideoCapture(source)
       
        # if not self.cap1.isOpened():
        #     raise RuntimeError('Failed to open camera.')

        #self.initSize(width, height)
        self.lmap, self.rmap = np.zeros((height, width//2, 2), dtype=np.float32), np.zeros((height, width//2, 2), dtype=np.float32)
        if maps is not None:
            if not os.path.exists(maps):
                raise FileNotFoundError('Maps\' folder not found.')
            else:
                lmaps, rmaps = [os.path.join(maps, f'lmap{i}') for i in [0, 1]], [os.path.join(maps, f'rmap{i}') for i in [0, 1]]
                self.lmap[..., 0], self.lmap[..., 1] = np.loadtxt(lmaps[0]), np.loadtxt(lmaps[1])
                self.rmap[..., 0], self.rmap[..., 1] = np.loadtxt(rmaps[0]), np.loadtxt(rmaps[1])
        else:
            print("1")
        #     logging.debug("No mapping file is loaded.")
        # logging.debug("BinocularCamera({}) initialization complete.".format(self.__name))
        
        set_fields_with_params()

    def initSize(self, width: int, height: int):
        self.w, self.h = width, height
        self.checkSize(width, height)

        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)

        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)


    def checkSize(self, width: int, height: int):
        supported = [(960, 960), (1280, 480), (1280, 720), (1280, 960), (2560, 720), (2560, 960),(1280,720)]
        if (width, height) not in supported:
            print(width,height)
            raise ValueError(f'({width}, {height}) pair not support.')

    def update(self,frame1: np.ndarray, frame2: np.ndarray ):
        """
        读取双目摄像头

        Params
        -----
        remap: bool = True
            是否消除畸变
        show_rectified: bool = False
            显示校正用的平行线
        """
        
        # ret1, frame1 = self.cap1.read()
        # ret2, frame2 = self.cap2.read()
        # frame1, frame2 = frame[:, :self.w//2, :], frame[:, self.w//2:, :]
        self.__raw = frame1, frame2
        # 图像校正
        frame1 = cv.undistort(frame1, camera_matrix1, distCoeff1)
        frame2 = cv.undistort(frame2, camera_matrix2, distCoeff2)
        self.__msg = frame1, frame2
        # t1 = time.time()
        if self.remap:
            # 立体校正
            frame1 = cv.remap(frame1, self.lmap, None, cv.INTER_LINEAR, cv.BORDER_CONSTANT, 0)
            frame2 = cv.remap(frame2, self.rmap, None, cv.INTER_LINEAR, cv.BORDER_CONSTANT, 0)
            self.__msg = frame1, frame2
            # 显示校正的平行线
        if self.show_rectified:
            for y in range(1, 21):
                pt1, pt2 = (0, frame1.shape[0]//20*y), (frame1.shape[1], frame1.shape[0]//20*y)
                cv.line(frame1, pt1, pt2, (0, 255, 0), 1)
                cv.line(frame2, pt1, pt2, (0, 255, 0), 1)
            self.__msg = frame1, frame2
            

    def read(self):
        """
        Returns
        -----
        left: np.ndarray
            经校正后的左图像
        right: np.ndarray
            经校正后的右图像
        """
        return BinocularCameraReturn(*self.__msg)

    def readRaw(self):
        """
        Returns
        -----
        left: np.ndarray
            原始的左图像
        right: np.ndarray
            原始的右图像
        """
        return BinocularCameraReturn(*self.__raw)

    def close(self):
        logging.debug("Closing BinocularCamera({})".format(self.__name))
        self.cap.release()
    
