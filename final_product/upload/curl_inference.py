
from posixpath import abspath
import time
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
#import logging
import argparse
import torch.optim as optim
import numpy as np
import datetime
import os.path
import os
import sys
# from torch.utils.tensorboard import SummaryWriter

from .data import Adobe5kDataLoader, Dataset
from . import metric
from . import model 


np.set_printoptions(threshold=sys.maxsize)


    


def inf_prep(dir_path) :
    
    dirpath = os.path.dirname(os.path.abspath(__file__))
    
    inf_path = dirpath + '/temp/images_inference.txt'
    
    inf_f = open(inf_path, "w")
    
    
    for filename in os.scandir(dir_path):
        if filename.is_file():
            inf_f.write(filename.name.split('.')[0] + '\n')
            

    inf_f.close()
    
    


def curl(inference_img_dirpath, 
         op_path,
         checkpoint_filepath = None):

    
    dirpath = os.path.dirname(os.path.abspath(__file__))

    if checkpoint_filepath == None :
        checkpoint_filepath = dirpath + '/pretrained_models/adobe_dpe/curl_validpsnr_23.073045286204017_validloss_0.0701291635632515_testpsnr_23.584083321292365_testloss_0.061363041400909424_epoch_510_model.pt'
   

    
    BATCH_SIZE=1  

    if (checkpoint_filepath is not None) and (inference_img_dirpath is not None):

        inf_prep(inference_img_dirpath)
        
        assert(BATCH_SIZE==1)

        temp_file_path =  dirpath + '/temp/images_inference.txt'
        
        inference_data_loader = Adobe5kDataLoader(data_dirpath=inference_img_dirpath,
                                                  img_ids_filepath=temp_file_path)
        inference_data_dict = inference_data_loader.load_data()
        inference_dataset = Dataset(data_dict=inference_data_dict,
                                    transform=transforms.Compose([transforms.ToTensor()]), normaliser=1,
                                    is_inference=True)

        inference_data_loader = torch.utils.data.DataLoader(inference_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                            num_workers=10)


        net = model.CURLNet()
        checkpoint = torch.load(checkpoint_filepath, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()

        criterion = model.CURLLoss()

        inference_evaluator = metric.Evaluator(
            criterion, inference_data_loader, "test", op_path)

        inference_evaluator.evaluate(net, epoch=0)



# curl('../test/eg_inp', '../test/img_op')