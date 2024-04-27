import torch
from model import load_model

with open('classnames') as f:
    CLASS_NAMES = f.read().split('\n')
    
if torch.cuda.is_available():
    DEVICE = torch.device('CUDA')
else:
    DEVICE = torch.device('cpu')

MODEL = load_model('weights\yolov3.weights', DEVICE)

