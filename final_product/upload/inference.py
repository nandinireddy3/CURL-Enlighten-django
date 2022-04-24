import matplotlib
matplotlib.use('agg')
from util import ImageProcessing
import matplotlib.pyplot as plt
from data import Adobe5kDataLoader, Dataset
import time
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import logging
import argparse
import torch.optim as optim
import numpy as np
import datetime
import os.path
import os
import metric
import model
import sys
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

def main() :
    FILE_NAME = './adobe5k_dpe/curl_example_test_input/a.png'
    MODEL = 'pretrained_models/adobe_dpe/curl_validpsnr_23.073045286204017_validloss_0.0701291635632515_testpsnr_23.584083321292365_testloss_0.061363041400909424_epoch_510_model.pt'
    DEVICE = 'cpu' if torch.is_available() else 'cpu'
    OUT_PATH = './adobe5k_dpe/curl_example_test_inference/'

    net = model.CURLNet()
    checkpoint = torch.load(MODEL, map_location=DEVICE)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    net.to(DEVICE)
    
    # img = cv2.imread(FILE_NAME)
    normaliser=2 ** 8 - 1
    
    img = ImageProcessing.normalise_image(np.array(Image.open(FILE_NAME).convert('RGB')), normaliser)
    
    transform=transforms.Compose([transforms.ToTensor()])
    
    
    
    t_img = transform(img).unsqueeze(0)
    t_img_2 = torch.clamp(t_img, 0, 1)
    
    t_img_2 = torch.Floattensor(t_img_2)
    
    # print(type(t_img_2))
    # print(t_img ==)
    
    with torch.no_grad():
        op ,_= net(t_img_2)
        print(type(op))
    




if __name__ == "__main__":
    main()