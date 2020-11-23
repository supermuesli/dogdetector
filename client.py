""" image classifier client. """

import os, sys, time

import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

from ml import CNN


def main():
	# if CUDA available, use it
	if torch.cuda.is_available():  
		dev = 'cuda:0' 
	else:  
		dev = 'cpu'  
	device = torch.device(dev) #

	# customize your datasource here
	dogs = '~/Downloads/dogsncats/dogs/1.jpg'

	# customize your CNN here
	model_path = 'autoencoder.ptc'

	with torch.no_grad():
		# create a CNN
		net = CNN(device=device)
		
		# load an existing model if possible
		net.load(model_path)
		net.eval() # inference mode

		#for p in net.parameters():
		#	print(p)

		# works with list of paths (which is more efficient with batches > 1)
		"""sample = net.transform([('D:\\dogsncats\\dogs\\' + '%d.jpg' % i) for i in range(8000, 9000)])
		
"""
		plt.ion()

		for i in range(10000):
			sample = ((torch.rand(64) - 0.5)*2).unsqueeze(0).cuda()
		
			grid = torchvision.utils.make_grid(net(sample).cpu(), nrow=100)
			
			plt.imshow(net.untransform(grid))
			plt.show()
			plt.pause(0.16)


if __name__ == '__main__':
	main()
