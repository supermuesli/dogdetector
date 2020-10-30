""" image classifier client. """

import os, sys

import torch
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
	model_path = 'model.asd'

	# create a CNN
	net = CNN()
	
	# load an existing model if possible
	net.load(model_path)

	if False:
		# works with a single path
		sample = net.transform(dogs)
		print(net(sample).mean())
	else:
		# works with list of paths (which is more efficient with batches > 1)
		sample = net.transform([('/home/kashim/Downloads/dogsncats/dogs/' + '%d.jpg' % i) for i in range(8000, 8665)])
		print('mean output dogs: ', net(sample).mean())

		sample = net.transform([('/home/kashim/Downloads/dogsncats/cats/' + '%d.jpg' % i) for i in range(8000, 8665)])
		print('mean output cats:', net(sample).mean())

		sample = net.transform([('/home/kashim/Downloads/dogsncats/cars/' + ('%d' % i).zfill(5)  + '.jpg') for i in range(8000, 8145)])
		print('mean output cars:', net(sample).mean())

if __name__ == '__main__':
	main()
