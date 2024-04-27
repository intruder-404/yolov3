import math, os, time, torch, PIL
from PIL import Image, ImageDraw
from IPython.display import Image as imshow

from utils import *
IMG_SIZE = 608

def plot_boxes(
    image,
    boxes,
    name=None,
    save_as=None
):
    width, height = image.width, image.height
    draw = ImageDraw.Draw(image)
    detections = []
    color = (0, 170, 210)
    
    for i in range(len(boxes)):
        box = boxes[i]
        x1, y1 = (box[0]-box[2]/2.0)*width, (box[1]-box[3]/2.0)*height
        x2, y2 = (box[0]+box[2]/2.0)*width, (box[1]+box[3]/2.0)*height
        if len(box) >= 7 and name:
            cls_conf = box[5]
            cls_id = box[6]
            detections += [(cls_conf, name[cls_id])]
            classes = len(name)
            offset = cls_id*123457 % classes
            draw.rectangle([x1, y1-15, x1+6.5*len(name[cls_id]), y1], fill=color)
            draw.text((x1+2, y1-13), name[cls_id], fill=(0,0,0))
            
        draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
        
    for (cls_conf, name_) in sorted(detections, reverse=True):
        print('%-10s: %f' %(name_, cls_conf))
    if save_as:
        print('Saving to.... %s'%save_as)
        image.save(save_as)
    
    return image

def bounding_boxes(
    model,
    conf_thresh,
    nms_thresh,
    image_dir,
    names,
    device,
    save_as=None
):
    assert os.path.exists(image_dir), 'IMAGE DOES NOT EXISTS'
    model.eval()
    image = Image.open(image_dir).convert('RGB')
    boxes = filter_boxes(
        model,
        device,
        image.resize((IMG_SIZE, IMG_SIZE)),
        conf_thresh,
        nms_thresh
    )
    predicted_image = plot_boxes(image, boxes, names, save_as)
    
    return predicted_image