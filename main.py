from multi_object_tracking import MOT, YOLOv7
from utils import plotBbox
import cv2 as cv
import mvsdk
from camera import Camera
from optdepth import locate
from inference import DepthModel
from transmissions.serial import Serial
import datetime
import torch
import time
from bincamera import BinocularCamera
import time
# from demo import draw

rate = 0.75

def cv_show(name,img):
    cv.imshow(name,img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    

# def resize(image, t:float = .5):
#     width, height = image.shape[:2]
#     return cv.resize(image,(int(height*t),int(width*t)), interpolation=cv.INTER_AREA)

# def read_data(fser):
#     # 先预设 data=0
#     data = 0
#     try:
#         # 读取一个字节，
#         temp = fser.read(1)
#     # IOError 更贴切，但未仔细实验，有时串口会因为某种原因产生 IO 问题，此时重启串口即可解决，此处正常应该添加一个循环，直到读取到数据为止
#     # 但最好设置最大循环次数防止死循环
#     except Exception as e:
#         print('nope')
#     return data

# def write_data(cls: int, fser, depth, angle, pd, pa):
#     # 修改串口输出的前缀，由具体要求而定
#     fser.setPrefix('AA')
#     # 10000 为深度值的最大值，为了归一化，65535 则对应 2^16 转换到 16 位的范围，返回值为成功发送的字节数
#     wr = fser.writeInIntEx(16, pd / 10000 * 65535, pa / 180 * 65535)
#     fser.setPrefix('FF')
#     return wr + fser.writeInIntEx(16, depth / 10000 * 65535, angle / 180 * 65535)
def resize(image, t: float = .5): 
    width, height = image.shape[:2]
    return cv.resize(image, (int(height*t), int(width*t)), interpolation=cv.INTER_AREA)

if __name__ == '__main__':
    

    depth_model = DepthModel('./Gwc-Net-light.pth')
    #深度模型
    s = Serial('d')
    s.start()
    #串口
    Ta = time.time()
    camera_1=Camera(0)
    camera_1.open("l")
    camera_2=Camera(1)
    camera_2.open("r")
    #相机
    #key = cv.waitKey(1) & 0xff
        
    # f1 = camera_2.read("l",1)
        
    # f2 = camera_1.read("r",0)
    c = BinocularCamera('c')
    c.start()
    
    #分类模型
    det = YOLOv7("./best_b.pt")
    det.start()
    # draw(666,888)
    Tb = time.time()
    print('加载相机及模型:',Tb-Ta)
    try:
        while 1:
            # print(6)
            T1=datetime.datetime.now()
            TQ = time.time()
            # key = cv.waitKey(1) & 0xff
            f1 = camera_2.read("l",0)
            # print(f1.width,f1.height)
            f2 = camera_1.read("r",1)
            # print(f1)
            # print("="*20)
            c.update(f1,f2)
            f1,f2=c.read()
            # print(f1)
            # print("="*20)

            #print(t1)
            # t2 = det(f2, classes=[0, 1, 2])

            #bboxes = det(f1,f2)[0]
            
            #bboxes = DetModel([resize(f1, t=.75)])[0]
            
            # print(bboxes.shape)
            # data=0
            
            data=s.read()

            print(data,type(data))
            
        
           
            
            # f1=resize(f1,t=.75)
            t1 = det(f1, data)
            bboxes = t1
            TW = time.time()
            print('识别时间:',TW-TQ)
            #s.write(1,1)
            # plotBbox(t1, f1)
            # plotBbox(t2, f2)

            if bboxes.numel()>0:
                TD = time.time()
                # bboxes[:, :4] = bboxes[:, :4] * 4/3
                # bboxes[:, :4]=bboxes[:, :4]*4/3
                (depth, angle), (pd, pa) = locate(data, bboxes, depth_model, f1, f2)
                depth = int(depth)
                angle = int(round(angle,1)*10)
                s.write(depth,angle)
                TF = time.time()
                # print('计算距离角度时间:',TF-TD)
                print("="*20)
                print(angle,depth)
                #draw(angle,depth)
                print("="*20)
                # target_number = bboxes.size(0)
            else:
                print('oopsy...nothing was detected!')
                # draw(0,0)
            
            # cv.imshow("0", f1)
            # cv.imshow("1", f2)
            TF = time.time()
            t2=datetime.datetime.now()
            print(TF-TQ)
            print(int(1/float(str(t2 - T1)[6:])))
            # if key == ord("q"): 
            #     break
        camera_2.close()
        camera_1.close()
    except :
        camera_2.close()
        camera_1.close()

    # cv.destroyAllWindows()



    # depth_model = DepthModel('./Gwc-Net-light.pth')
    # f1 = cv.imread('ltt2.jpg')
    # f2 = cv.imread('rtt2.jpg')
    # det = YOLOv7('./best_b.pt')
    # det.start()
    # t1 = det(f1, classes=[0])
    # t2 = det(f2, classes=[0])
    # bboxes = [f1, f2]
    # data = 0
    # plotBbox(t1, f1)
    # plotBbox(t2, f2)
    # # if f1.numel() > 0:
    # #     (depth, angle), (pd, pa) = locate(data, bboxes, depth_model, f1, f2)
    
    # #     target_number = bboxes.size(0)
    # # else:
    # #     print('oopsy...nothing was detected!')
    # cv_show("p1", f1)
    # cv_show("p2", f2)
