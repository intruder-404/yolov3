
import torch 
from torch import nn
from utils import *

def iou(box1, box2, x1y1x2y2 = True):
    
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        
        w1, h1 = box1[2]-box1[0], box1[3]-box1[1]
        w2, h2 = box2[2]-box2[0], box2[3]-box2[1]
        
    else:
        mx = min(box1[0] -box1[2]/2, box2[0] -box2[2]/2)
        Mx = max(box1[0] +box1[2]/2, box2[0] +box2[2]/2)
        my = min(box1[1] -box1[3]/2, box2[1] -box2[3]/2)
        My = max(box1[1] +box1[3]/2, box2[1] +box2[3]/2)
        
        w1, h1 = box1[2], box1[3]
        w2, h2 = box2[2], box2[3]
        
    uw, uh = Mx - mx, My - my
    cw, ch = w1 + w2 - uw, h1 + h2 - uh
    
    intersection = 0
    
    if cw <= 0 and ch <= 0:
        return .0
    
    area1 = w1 * h1
    area2 = w2 * h2
    intersection = cw * ch
    union = area1 + area2 - intersection
    return intersection/union


def non_max_supression(
    boxes,
    nms_thresh
):
    if len(boxes) == 0:
        return boxes
    
    det_confs = torch.zeros(len(boxes))
    
    for i in range(len(boxes)):
        det_confs[i] = 1 - boxes[i][4]
        
    _, ids_sorted = torch.sort(det_confs)
    out_boxes = []

    for i in range(len(boxes)):
        box_i = boxes[ids_sorted[i]]
        if box_i[4]>0:
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)):
                box_j = boxes[ids_sorted[j]]
                if iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    box_j[4] = 0
    return out_boxes