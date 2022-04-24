# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of the CVPR 2020 paper:
"Deep Local Parametric Filters for Image Enhancement": https://arxiv.org/abs/2003.13985

Please cite the paper if you use this code

Tested with Pytorch 0.3.1, Python 3.5

Authors: Sean Moran (sean.j.moran@gmail.com), 
		 Pierre Marza (pierre.marza@gmail.com)

Instructions:

To get this code working on your system / problem you will need to edit the
data loading functions, as follows:

1. main.py, change the paths for the data directories to point to your data
directory (anything with "/aiml/data")

2. py, lines 216, 224, change the folder names of the data input and
output directories to point to your folder names
'''
import model
from data import get_loader
# ---------------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import glob
import math
import numpy
import torch
import copy
import time
import logging
import argparse
import datetime
import skimage
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor
numpy.set_printoptions(threshold=sys.maxsize)
import warnings
warnings.filterwarnings("ignore")
print(torch.__version__)


# some useful functions
to_numpy = lambda _image: _image.data.mul_(255).clamp_(0, 255).permute(1, 2, 0).cpu().numpy().astype('uint8')

# 将图像 _image 保存到 file_name
def back_to_image(_image, file_name):
	ndarr = to_numpy(_image)
	im = Image.fromarray(ndarr)
	im.save(file_name)




def main():

	parser = argparse.ArgumentParser(
		description="Train the DeepLPF neural network on image pairs")
	parser.add_argument("--regen",  help="Regen images", action='store_true')
	parser.add_argument("--train",  help="Train the model", action='store_true')
	parser.add_argument("--test",  help="test the model", action='store_true')
	# for train
	parser.add_argument(
		"--train_dir", default="./datasets/train/", type=str, required=False, help="Directory of training images")
	parser.add_argument(
		"--train_images_list", default="./datasets/train_images_list.txt", type=str, required=False, help="Directory of training images")
	parser.add_argument(
		"--valid_dir", default="./datasets/test/", type=str, required=False, help="Directory of training images")
	parser.add_argument(
		"--valid_images_list", default="./datasets/test_images_list.txt", type=str, required=False, help="Directory of training images")
	parser.add_argument(
		"--num_epoch", default=30, type=int, required=False, help="Number of epoches (default 5000)")
	parser.add_argument(
		"--valid_every", default=1, type=int, required=False, help="Number of epoches after which to compute validation accuracy")
	# for application
	parser.add_argument(
		"-c", "--checkpoint_filepath", default='./checkpoints/log2020-09-11_15-06-47/curl_validpsnr_25.092590592669_validloss_0.022607844322919846_epoch_30_model.pt', required=False, help="Location of checkpoint file")
	parser.add_argument(
		"-i", "--input_dir", default='./datasets/test/input/', required=False, help="Directory containing images to run through a saved DeepLPF model instance")
	parser.add_argument(
		"-l", "--label_dir", default='./datasets/test/expertC_gt/', required=False, help="Directory containing high quality images")
	parser.add_argument(
		"-o", "--output_dir", default='./results/', required=False, help="To save enhanced images")

	args = parser.parse_args()

	if(args.regen or args.test):
		assert args.checkpoint_filepath is not None and os.path.exists(args.checkpoint_filepath), \
			"Please check the checkpoint_filepath!"
		if(not os.path.exists(args.input_dir)):
			os.makedirs(args.input_dir)
		if(not os.path.exists(args.output_dir)):
			os.makedirs(args.output_dir)

	# ---------------- judge whether to enhance images only
	if(args.regen):
		# prepare dataset
		inference_data_loader = get_loader(
			input_dir=args.input_dir, 
			for_test=True, 
			batch_size=1, 
			shuffle=False, 
			num_workers=4, 
			normaliser=1,
			img_ids_filepath=None
		)
		
		# load ttrained networks
		net = torch.load(args.checkpoint_filepath, map_location='cpu')
		net.to("cpu")
		net.eval()

		for batch_num, data in enumerate(inference_data_loader):
			# 加载数据, 送到 GPU
			image_low_quality, image_name = Variable(data['input']).to("cpu"), data['name']
			print('{}/{}===>  processing {}'.format(batch_num + 1, len(inference_data_loader), image_name))
			# 增强图像
			res, _ = net(image_low_quality)
			new_image = res.squeeze(0)
			# 保存
			back_to_image(new_image, args.output_dir + image_name[0])



	elif(args.test):

		# prepare dataset
		inference_data_loader = get_loader(
			input_dir=args.input_dir, 
			for_test=True, 
			batch_size=1, 
			shuffle=False, 
			num_workers=4, 
			normaliser=1,
			img_ids_filepath=None
		)

		label_data_loader = get_loader(
			input_dir=args.label_dir, 
			for_test=True, 
			batch_size=1, 
			shuffle=False, 
			num_workers=4, 
			normaliser=1,
			img_ids_filepath=None
		)
		
		# load ttrained networks
		net = torch.load(args.checkpoint_filepath, map_location='cpu')
		net.to("cpu")
		net.eval()
		
		mean_psnr = 0
		mean_ssim = 0
		for batch_num, (low_quality, high_quality) in enumerate(zip(inference_data_loader, label_data_loader)):
			# 加载数据, 送到 GPU
			low_quality_image, low_image_name = Variable(low_quality['input']).to("cpu"), low_quality['name']
			high_quality_image, high_image_name = Variable(high_quality['input']).to("cpu"), high_quality['name']
			assert(low_image_name == high_image_name)
			# get enhanced result
			res, _ = net(low_quality_image)
			# trans to numpy data
			res = to_numpy(res.squeeze(0))
			high_quality_image = to_numpy(high_quality_image.squeeze(0))
			# psnr | ssim
			psnr_value = skimage.measure.compare_psnr(res, high_quality_image)
			ssim_value = skimage.measure.compare_ssim(res, high_quality_image, multichannel=True)

			print('{}/{}===>  PSNR {} | SSIM {} | {}'.format(
				batch_num + 1, len(inference_data_loader), psnr_value, ssim_value, low_image_name))

			mean_psnr += psnr_value
			mean_ssim += ssim_value
		mean_psnr /= (batch_num + 1)
		mean_ssim /= (batch_num + 1)
		print('test phase  :  PSNR {}  |  SSIM  {}'.format(mean_psnr, mean_ssim))



	elif(args.train):

		# ---------------------- some important parameters
		num_epoch = args.num_epoch
		valid_every = args.valid_every

		# ---------------------- prepare log
		timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
		log_dirpath = "./checkpoints/log" + timestamp
		os.mkdir(log_dirpath)
		handlers = [logging.FileHandler(
			log_dirpath + "/deep_lpf.log"), logging.StreamHandler()]
		logging.basicConfig(
			level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=handlers)
		logging.info('######### Parameters #########')
		logging.info('Number of epochs: ' + str(num_epoch))
		logging.info('Logging directory: ' + str(log_dirpath))
		logging.info('Dump validation accuracy every: ' + str(valid_every))
		logging.info('##############################')

		# ---------------------- prepare datasets for training and validing
		training_data_loader = get_loader(
			input_dir=args.train_dir, 
			for_test=False, 
			batch_size=1, 
			shuffle=True, 
			num_workers=4, 
			normaliser=1,
			img_ids_filepath=args.train_images_list
		)
		validation_data_loader = get_loader(
			input_dir=args.valid_dir, 
			for_test=False, 
			batch_size=1, 
			shuffle=False, 
			num_workers=4, 
			normaliser=1,
			img_ids_filepath=args.valid_images_list
		)

		# ---------------------- prepare network
		net = model.CURLNet()

		logging.info('######### Network created #########')

		# loss functions
		criterion = model.CURLLoss(ssim_window_size=5)
		# optimizer
		optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)
		best_valid_psnr = 0.0

		# ---------------------- training phase
		optimizer.zero_grad()
		net.train()
		net.to("cpu")

		running_loss = 0.0
		examples = 0
		psnr_avg = 0.0
		ssim_avg = 0.0
		batch_size = 1
		# train for num_epoch
		for epoch in range(num_epoch):

			examples = 0.0
			running_loss = 0.0
			
			for batch_num, data in enumerate(training_data_loader, 0):
				# load images
				input_img_batch, output_img_batch, category = Variable(data['input'], requires_grad=False).to("cpu"), Variable(data['expertC_gt'], requires_grad=False).to("cpu"), data['name']
				# forward phase
				net_output_img_batch, gradient_regulariser = net(
					input_img_batch)
				net_output_img_batch = torch.clamp(net_output_img_batch, 0.0, 1.0)

				# compute the loss
				loss = criterion(net_output_img_batch, output_img_batch, gradient_regulariser)

				# backward to optimize the loss
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				# output some information
				running_loss += loss.data[0]
				examples += batch_size

				if((batch_num + 1) % 1000 == 0):
					break

				print('epoch {} | {}/{}==>  loss  :  {}'.format(
					epoch, batch_num + 1, 5000, running_loss.cpu()[0] / examples))
			logging.info('[%d] train loss: %.15f' %
						 (epoch + 1, running_loss / examples))
			print('{}==>  train_loss  :  {}'.format(epoch + 1, running_loss.cpu()[0] / examples))


			if((epoch + 1) % valid_every == 0):
				# -------------------------------------------------------------------------
				# valid phase every epoch
				examples = 0.0
				running_loss = 0.0

				mean_psnr = 0
				mean_ssim = 0

				for batch_num, data in enumerate(validation_data_loader, 0):

					net.eval()
					# load images for validation
					input_img_batch, output_img_batch, category = Variable(data['input'], requires_grad=False).to("cpu"), Variable(data['expertC_gt'], requires_grad=False).to("cpu"), data['name']

					net_output_img_batch, gradient_regulariser  = net(
						input_img_batch)
					net_output_img_batch = torch.clamp(net_output_img_batch, 0.0, 1.0)

					optimizer.zero_grad()

					loss = criterion(net_output_img_batch, output_img_batch, gradient_regulariser)

					running_loss += loss.data[0]
					examples += batch_size

					# trans to numpy data
					res = to_numpy(net_output_img_batch.squeeze(0))
					high_quality_image = to_numpy(output_img_batch.squeeze(0))
					# psnr | ssim
					psnr_value = skimage.measure.compare_psnr(res, high_quality_image)
					ssim_value = skimage.measure.compare_ssim(res, high_quality_image, multichannel=True)

					print('valid {}/{}  PSNR {}  |  SSIM {}'.format(batch_num, len(validation_data_loader), psnr_value, ssim_value))
					mean_psnr += psnr_value
					mean_ssim += ssim_value

					if(batch_num % 10 == 0):
						_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
						new_image = torch.cat([input_img_batch, output_img_batch, net_output_img_batch], dim=-1).squeeze(0)
						back_to_image(new_image, './for_valid_phase/epoch_' + str(epoch) + '_' + str(batch_num) + '_' + _timestamp + '.png')
				
				mean_psnr /= (batch_num + 1)
				mean_ssim /= (batch_num + 1)

				torch.save(net, os.path.join(log_dirpath, str(epoch) + '_PSNR_' + str(mean_psnr) \
					+ '_SSIM_' + str(mean_ssim) + '.pth'))

				logging.info('[%d] valid loss: %.15f  |  PSNR: %.15f  |  SSIM: %.15f' %
							 (epoch + 1, running_loss / examples, mean_psnr, mean_ssim))
				print('{}==>  valid_loss  :  {}  |  PSNR  :  {}  |  SSIM  :  {}'.format(
					epoch + 1, running_loss.cpu()[0] / examples, mean_psnr, mean_ssim))
				# back to train mode
				net.train()



if __name__ == "__main__":
	main()