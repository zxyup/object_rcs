from math import ceil, floor
#from attr 
import has
import cv2 as cv
from camera import focus, base, u01, u02, dx_f
import numpy as np
import torch
h = 0
import matplotlib.pyplot as plt
def convert_int(*args):
    """将args内的参数转换为整型, 推荐加个异常捕获模块 try... except...，如果这类工具函数比较多可以直接通过装饰器来添加"""
    return [int(arg) for arg in args]

def clip(x, y):
    """通过截断控制 x, y 取值范围，不够通用，仅适用于480x640的图像"""
    x = max(min(x, 480), 0)
    y = max(min(y, 640), 0)
    return x, y

def getCenter(pred: torch.Tensor):
    """获取检测框中心点"""
    N = pred.size(0)
    center = torch.zeros((N, 2), dtype=pred.dtype)
    center[:, 0] = (pred[:, 0] + pred[:, 2])/2
    center[:, 1] = (pred[:, 1] + pred[:, 3])/2
    return center

def getLocationFromDepth(depth, pred: torch.Tensor):
    center = getCenter(pred)
    b_ = ((center[:, 0] - u01) * dx_f) * depth
    factor = 180/np.pi
    angle = torch.arctan(depth/(base - b_)) * factor
    return depth, -angle

def get_depth(bbox, disp):
    n = 1
    depth = np.zeros(bbox.size(0), dtype=np.float32)
    for i in range(bbox.size(0)):
        x1, y1, x2, y2 = bbox[i]
        width, height = y2-y1, x2-x1
        y2, y1, x2, x1 = y2 - width * 0.2, y1 + width * 0.2, x2 - height * 0.2, x1 + height * 0.2
        x1, y1, x2, y2 = convert_int(x1, y1, x2, y2)
        x1, y1 = clip(x1, y1)
        x2, y2 = clip(x2, y2)
        dbox = disp.new_empty((y2-y1, x2-x1))
        # size = disp.size()
        # logger.variable(x1, x2, y1, y2, size)
        dbox.copy_(disp[y1:y2, x1:x2])
        median = dbox.median()
        mad = (dbox - median).median()
        dbox[dbox > median + n * mad] = median + n * mad
        dbox[dbox < median - n * mad] = median - n * mad
        depth[i] = dbox.mean()
    return focus * base / depth

def _padding(v1, v2, mod: int = 32):
    pad = mod - ((v2 - v1) % mod)
    if v1 < mod:
        v2 += pad
    else:
        v1 -= pad
    return v1, v2

def _pick_disp(cx, cy, disp, ux, ly, scale: int = 3):
    cx -= ux
    cy -= ly
    h, w = disp.size()
    t_disp = torch.zeros(cx.size())
    for i in range(cx.size(0)):
        l, r, u, b = convert_int(max(0, cx[i]-scale), min(w, cx[i]+scale), max(0, cy[i]-scale), min(h, cy[i]+scale))
        t_disp[i] = disp[u:b, l:r].mean()
        # cv.rectangle(left, (l+ux-10, u+ly-20), (l+ux+100, u+ly-8), color=(96, 96, 96), thickness=-1)
        # cv.putText(left, f"depth-{focus * base / t_disp[i]/1000:.2f}m", (l+ux-10, u+ly-8),
        # fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(255, 255, 255), thickness=1)
        # cv.rectangle(left, (l+ux, u+ly), (r+ux, b+ly), (0, 255, 0), thickness=5)
        # cv.rectangle(left, (l+ux, u+ly-h), (r+ux, b+ly-h), (0, 255, 0), thickness=5)
    return t_disp

def get_depth_optimized(bbox, depth_module, left, right, max_disp: int = 192, scale: float = .25, shrinking: bool = False):
    """简单优化过的深度获取，只将一个能囊括所有检测框的部分中心至其左边的 `max_disp` 范围内的图像送入深度模型，然后获取一个更小范围的深度值
    scale 与 shrink 参数相辅，用于将囊括所有检测框的大框在放缩，贸然缩小可能会导致将小物体的数据丢失。
    
    参数
    -----
    bbox: torch.Tensor
        为对应锚框的数据 [x1,y1,x2,y2,..]
    left, right: numpy.ndarray
        左右视图的图像
    max_disp: int
        深度估计模型的最大视差
    
    返回
    -----
    depth: numpy.ndarray
        对应检测框内的深度图
    """
    # print(left.shape,right.shape)
    ux, lx = torch.min(bbox[:, 0]).item(), torch.max(bbox[:, 2]).item()
    ly, ry = torch.min(bbox[:, 1]).item(), torch.max(bbox[:, 3]).item()
    cx = torch.div(bbox[:, 0] + bbox[:, 2], 2, rounding_mode='trunc')
    cy = torch.div(bbox[:, 1] + bbox[:, 3], 2, rounding_mode='trunc')
    # shrinking
    if shrinking:
        shrink = (ry - ly) * scale
        ly = floor(ly + shrink)
        ry = ceil(ry - shrink)
    # add max_disp
    ux = max(0, ux - max_disp)
    # 由于深度估计模型的输入到输出一共经过 32 倍 (应该是) 的下采样 ，故图像应该为该倍数才不会在卷积的运算中出现矩阵不匹配的情况
    ux, lx = _padding(ux, lx)
    ly, ry = _padding(ly, ry)
    ly, ry, ux, lx = convert_int(ly, ry, ux, lx)
    # print(ly,ry,ux,lx)
    # 传入部分图像获取视差，此时获得的视差图也是部分    图像的视差
    disp = depth_module(left[ly:ry, ux:lx], right[ly:ry, ux:lx])
    # vis
    #show = np.uint8((disp / max_disp * 255).detach().cpu().numpy())
    #show = cv.cvtColor(show, cv.COLOR_GRAY2BGR)
    #print(show.ndim)
    global h
    h = lx - ux
    w = ry - ly
    # left[ly-w:ry-w, ux:lx] = disp
    
    # cv.rectangle(left, (ux, ly), (lx, ry), (255, 0, 0), thickness=2)
    # cv.rectangle(left, (ux, ly-w), (lx, ry-w), (255, 0, 0), thickness=2)
    # cv.namedWindow('disp')
    # disp=disp/255
    # plt.imshow(disp.numpy())
    # plt.imshow()
    # print(disp)
    # print(type(disp))
    # cv.imshow('disp',disp.numpy())
    # cv.waitKey(0)
    

    # 计算公式 f*b/Z, 其中 f 为相机焦距，b 为基准面到达的距离，Z 为视差值
    return focus * base / _pick_disp(cx, cy, disp, ux, ly) 

def getNearestTarget(depth: torch.Tensor, angle: torch.Tensor):
    """
    获取最近的目标

    参数
    -----
    depth: ndarray
        深度信息
    angle: ndarray
        角度信息, 弧度制

    返回
    -----
    最近的(depth, angle)，角度制
    """
    if depth.numel() == 0:
        return (0, 0)
    idx = depth.argmin()
    depth = depth[idx]
    angle = angle[idx] if angle[idx] > 0 else 180 + angle[idx]
    angle = angle/180 * np.pi
    angle = torch.arctan(depth/(depth/torch.tan(angle)+60))/np.pi * 180
    angle = angle if angle > 0 else 180 + angle 
    
    return depth.item(), angle.item()

def locate(cls: int, bbox, depth_model, left, right):
    """仅供参考"""
    #class_dict ={0:0, 1:1 ,2:2}
    has_pillar = False
    #cls = class_dict.get(cls, [])
    mask = (bbox[:, -1].int() == 4)
    pidx = 0
    if mask.any():
        has_pillar = True
        for i in range(mask.size(0)):
            if mask[i]:
                pidx = i
                break

    mask = mask | (bbox[:, -1].int() == cls)
    tnum = 0
    for i in range(mask.size(0)):
        if i == pidx:
            pidx = tnum
            break
        if mask[i]:
            tnum += 1
    res = bbox# [mask] # => res = bbox
    idx = [i for i in range(res.size(0))]
    if has_pillar:
        idx.remove(pidx)
    if res.numel() > 0:
        depth = get_depth_optimized(res[:, :4], depth_model, left, right)
        depth, angle = getLocationFromDepth(depth, res[:, :4])
        if has_pillar: 
            pd = depth[pidx]
            pa = angle[pidx]
        else: pd = pa = 0
        return getNearestTarget(depth[idx], angle[idx]), (pd, pa)
    else:
        return (0, 0), (0, 0)
    
    
