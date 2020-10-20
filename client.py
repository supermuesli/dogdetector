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
	dogs = '/home/muesli/Downloads/dogscats/dogs'

	# customize your CNN here
	model_path = 'model.asd'

	# create a CNN
	net = CNN()
	# load an existing model if possible
	
	net.load(model_path)

	# works with list of paths (more efficient with batches)
	sample = net.transform([(dogs + '/%d.jpg' % i) for i in range(100)])
	print(net(sample))

	# works with a single path
	sample = net.transform(dogs + '/3.jpg')
	print(net(sample))

if __name__ == '__main__':
	main()
