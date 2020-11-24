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

	# customize your datasource here
	dogs = '~/Downloads/dogsncats/dogs/1.jpg'

	# customize your CNN here
	model_path = 'autoencoder.pth'

	with torch.no_grad():
		# create a CNN
		net = CNN()
		
		# load an existing model if possible
		net.load(model_path)
		net.eval() # inference mode

		#for p in net.parameters():
		#	print(p)

		# works with list of paths (which is more efficient with batches > 1)
		"""
		sample = net.transform([('/home/kashim/Downloads/dogsncats/cars/' + ('%d' % i).zfill(5) + '.jpg' ) for i in range(1, 10)])
		#sample = net.transform(['/home/kashim/Desktop/niggtri.jpg'	, '/home/kashim/Downloads/dogsncats/dogs/4.jpg'])
		e = net.encode(sample)

		b = torch.tensor([])

		plt.ion()

		m = 0
		n = 20
		s = 1
		while True:
			for i in range(m, n, s):
				#b = torch.cat((b, (i/n * e[0] + (1-i/n) * e[1]).unsqueeze(0)))
				
				#grid = torchvision.utils.make_grid(net.decode(e[0] + (i/abs(m-n) * e[1]).unsqueeze(0)), nrow=1)
				grid = torchvision.utils.make_grid(net.decode(e[i].unsqueeze(0)), nrow=1)
				
				plt.imshow(net.untransform(grid))
				plt.show()
				plt.pause(2)
				
			m, n = n, m
			s *= -1



		"""
		
		#sample = ((torch.tensor([i/1000 for j in range(128)]) - 0.5)*2).unsqueeze(0)
		sample = net.transform([('/home/kashim/Downloads/dogsncats/dogs/' + ('%d' % i) + '.jpg' ) for i in range(9000, 9500)])

		grid = torchvision.utils.make_grid(net(sample), nrow=16)
		
		plt.imshow(net.untransform(grid))
		plt.show()
		
if __name__ == '__main__':
	main()
