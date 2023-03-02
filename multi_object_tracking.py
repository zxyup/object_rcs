import os
from typing import Sequence

import cv2
import torch
import numpy as np
import cv2 as cv
import time
from torch.backends import cudnn
from typing import List
import torch.nn as nn

from sort import Sort
from utils import attempt_load, non_max_suppression, scale_coords, letterbox, plotBbox


cudnn.benchmark = True


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)
    
    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
        print("None")
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv.resize(image, tuple(patch_shape[::-1]))
    return image




class YOLOv7(nn.Module):
    
    #def __init__(self, model_path: os.PathLike, cpu: bool = False):
    def __init__(self, model_path: os.PathLike):
        super().__init__()
        self.model_path = model_path
        self.device = torch.device('cuda:0')
       # self.device = torch.device('cuda:0' if not cpu else 'cpu')
        #self.half = not cpu
        
        
    def start(self, img_size: int = 640, stride: int = 32):
        self.img_size = img_size
        self.stride = stride
        # print('1')
        self.model = attempt_load(self.model_path, map_location=self.device)
        # print('2')
        self.model.eval()
        # print('3')
        if self.half:
            self.model.half()
            
    def forward(self, 
        img: np.ndarray,
        classes: Sequence[int] ,
        conf_thres: float = .5,
        iou_thres: float = .45,
        
        agnostic_nms: bool = False
        ):
        # Preprocess
        shape = img.shape
        img = letterbox(img, self.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416\
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        # ret = [torch.empty(0)] * len(img)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():
            pred = self.model(img, augment=False)[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=agnostic_nms, classes=classes)[0]
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], shape).round()
        return pred 


    def forward_batches(self, 
        imgs: List[np.ndarray],
        conf_thres: float = .25,
        iou_thres: float = .45,
        classes: Sequence[int] = [0],
        agnostic_nms: bool = False
        ):
        shape = imgs[0].shape
        imgs = [letterbox(img, self.img_size, stride=self.stride)[0] for img in imgs]
        imgs = [img[:, :, ::-1].transpose(2, 0, 1) for img in imgs]  # BGR to RGB, to 3x416x416\
        imgs = np.array(imgs)
        imgs = torch.from_numpy(imgs).to(self.device)
        imgs = imgs.half() if self.half else imgs.float()
        imgs /= 255.0

        with torch.no_grad():
            preds = self.model(imgs, augment=False)[0]
            # nms
            preds = non_max_suppression(preds, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
            for i in range(len(preds)):
                preds[i][:, :4] = scale_coords(imgs.shape[2:], preds[i][:, :4], shape).round()
        return preds


class MOT(object):

    def __init__(self, yolo_model: os.PathLike, cpu: bool = False):
        self.yolo_model = YOLOv7(yolo_model, cpu)
        self.device = torch.device('cuda:0' if not cpu else 'cpu')   
        MOT.GLOBAL_MOT = self 

    def start(self, img_size: int = 640):
        self.yolo_model.start(img_size)
        

    def run_with_sort(self, video: cv.VideoCapture, max_age: int = 20, output: os.PathLike = None):
        """
        Running YOLOv7 and SORT
        """
        tracking_results = {}
        self.tracker = Sort(max_age)
        if output is not None:
            fourcc = cv.VideoWriter_fourcc(*'XVID') 
            fps = int(video.get(cv.CAP_PROP_FPS))
            shape = video.get(cv.CAP_PROP_FRAME_WIDTH), video.get(cv.CAP_PROP_FRAME_HEIGHT)
            shape = tuple(map(int, shape))

            video_writer = cv.VideoWriter(output, fourcc, fps, shape)
        i_frame = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            det = self.yolo_model.forward(frame)
            # Update tracker.
            if det.shape[0] > 0:
                bboxes = self.tracker.update(det.cpu())
                for box in bboxes:
                    person_dict = tracking_results.setdefault(int(box[-1]), 
                        {'bbox': [], 'frames': []})
                    person_dict['bbox'].append(box)
                    person_dict['frames'].append(i_frame)
            if output is not None:
                plotBbox(bboxes, frame)
                video_writer.write(frame)
            i_frame += 1
        for pid in tracking_results.keys():
          tracking_results[pid]['bbox'] = np.asarray(tracking_results[pid]['bbox'])
          tracking_results[pid]['frames'] = np.asarray(tracking_results[pid]['frames'])
        if output is not None:
            video_writer.release()
        return tracking_results
        
    def run_with_sort_batches(self, video: cv.VideoCapture, max_age: int = 20, batches: int = 16, output: os.PathLike = None):
        """
        Running YOLOv7 and SORT
        """
        if output is not None:
            fourcc = cv.VideoWriter_fourcc(*'XVID') 
            fps = int(video.get(cv.CAP_PROP_FPS))
            shape = video.get(cv.CAP_PROP_FRAME_WIDTH), video.get(cv.CAP_PROP_FRAME_HEIGHT)
            shape = tuple(map(int, shape))

            video_writer = cv.VideoWriter(output, fourcc, fps, shape)
        tracking_results = {}
        self.tracker = Sort(max_age)
        i_frame = 0
        one_batch = []
        valid = True
        while valid:
            valid, frame = video.read()
            if valid:
                one_batch.append(frame)
            # empty batch
            if len(one_batch) == 0:
                break
            if len(one_batch) == batches or (not valid and len(one_batch) > 0):
                bat_num = len(one_batch)
                dets = self.yolo_model.forward_batches(one_batch)
                for i in range(bat_num):
                    det = dets[i]
                    # Update tracker.
                    if det.shape[0] > 0:
                        bboxes = self.tracker.update(det.cpu())
                        for box in bboxes:
                            person_dict = tracking_results.setdefault(int(box[-1]), 
                                {'bbox': [], 'frames': []})
                            person_dict['bbox'].append(box[:4])
                            person_dict['frames'].append(i_frame - bat_num + i + 1)
                        if output is not None:
                            cf = one_batch[i]
                            plotBbox(bboxes, cf)
                            video_writer.write(cf)
                one_batch.clear()
            i_frame += 1
        if output is not None:
            video_writer.release()
        return tracking_results

if __name__ == '__main__':
    det = YOLOv7('./yolov7.pt', True)
    video = cv2.VideoCapture(0)
    img = cv.imread('./test_img.jpg')
    det.start()
    res = det(img, classes=[0, 2, 3, 7])
    plotBbox(res, img)
    cv.imshow("p", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
        

