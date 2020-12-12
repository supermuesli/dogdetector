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
#from bhtsne import tsne

from classifier import DogDetector
from ml import GrayVAE

def main():

	# customize your datasource here
	dogs = '~/Downloads/dogsncats/dogs/1.jpg'


	with torch.no_grad():
		# create a GrayVAE
		#vae = GrayVAE()
		dd = DogDetector()

		dd.load('classifier.pth')
		dd.eval()

		# load an existing model if possible
		# vae.load('autoencoder.pth')
		# vae.eval() # inference mode

		dogs = dd.transform([('D:\\dogsncats\\dogs\\' + ('%d' % i) + '.jpg' ) for i in range(8000, 8100)])
		cats = dd.transform([('D:\\dogsncats\\cats\\' + ('%d' % i) + '.jpg' ) for i in range(8000, 8100)])
		
		#encoded_dogs = vae.encode(dogs)
		#encoded_cats = vae.encode(cats)
		#encodings = torch.cat((encoded_dogs, encoded_cats)).double()
		#print(encodings.shape)
		#Y = tsne(encodings)

		#plt.scatter(Y[:, 0], Y[:, 1], c=['blue']*100 + ['red']*100)
		#plt.show()


		# grid = torchvision.utils.make_grid(vae(dogs), nrow=8)
		# plt.imshow(vae.untransform(grid.cpu()))
		# plt.show()

		# grid = torchvision.utils.make_grid(vae(cats), nrow=8)
		# plt.imshow(vae.untransform(grid.cpu()))
		# plt.show()

		
		print('dogs:', dd(dogs).mean())
		
		print('cats:', dd(cats).mean())
		
		
		"""


		# works with list of paths (which is more efficient with batches > 1)
		sample = dd.transform([('/home/kashim/Downloads/dogsncats/cars/' + ('%d' % i).zfill(5) + '.jpg' ) for i in range(1, 10)])
		#sample = dd.transform(['/home/kashim/Desktop/niggtri.jpg'	, '/home/kashim/Downloads/dogsncats/dogs/4.jpg'])
		e = vae.encode(sample)

		b = torch.tensor([])

		plt.ion()

		m = 0
		n = 20
		s = 1
		while True:
			for i in range(m, n, s):
				#b = torch.cat((b, (i/n * e[0] + (1-i/n) * e[1]).unsqueeze(0)))
				
				#grid = torchvision.utils.make_grid(vae.decode(e[0] + (i/abs(m-n) * e[1]).unsqueeze(0)), nrow=1)
				grid = torchvision.utils.make_grid(vae.decode(e[i].unsqueeze(0)), nrow=1)
				
				plt.imshow(vae.untransform(grid))
				plt.show()
				plt.pause(2)
				
			m, n = n, m
			s *= -1



		
		#sample = ((torch.tensor([i/1000 for j in range(128)]) - 0.5)*2).unsqueeze(0)
		sample = vae.transform([('/home/kashim/Downloads/dogsncats/dogs/' + ('%d' % i) + '.jpg' ) for i in range(9000, 9500)])

		grid = torchvision.utils.make_grid(vae(sample), nrow=16)
		
		plt.imshow(vae.untransform(grid))
		plt.show()
		"""
		
if __name__ == '__main__':
	main()
