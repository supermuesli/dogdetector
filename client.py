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
	model_path = 'model.ptc'

	with torch.no_grad():
		# create a CNN
		net = CNN()
		
		# load an existing model if possible
		net.load(model_path)
		net.eval() # inference mode

		#for p in net.parameters():
		#	print(p)

		# works with list of paths (which is more efficient with batches > 1)
		sample = net.transform([('/home/muesli/Downloads/dogsncats/dogs/' + '%d.jpg' % i) for i in range(0, 100)])
		for t in net(sample):
			im = net.untransform(t)
			im.show()
			input('press enter to continue')

if __name__ == '__main__':
	main()
