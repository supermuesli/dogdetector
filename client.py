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
	dogs = '/home/kashim/Downloads/dogsncats/dogs'

	cats = '/home/kashim/Downloads/dogsncats/cats'

	# customize your CNN here
	model_path = 'model.asd'

	# create a CNN
	net = CNN()
	
	# load an existing model if possible
	net.load(model_path)

	if sys.argv[1:] != []:
		# works with a single path
		print(sys.argv[1:])
		sample = net.transform(sys.argv[1:])
		print(net(sample).mean())
	else:
		# works with list of paths (more efficient with batches)
		sample = net.transform([(dogs + '/%d.jpg' % i) for i in range(8000, 8665)])
		print('mean sensitivity: ', net(sample).mean())


		sample = net.transform([(cats + '/%d.jpg' % i) for i in range(8000, 8665)])
		print('mean type I error: ', net(sample).mean())

if __name__ == '__main__':
	main()
