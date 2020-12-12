""" universal grayscale image classifier. see https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html """

import os, logging, random, sys, math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

from ml import GrayVAE

class ImageGrayScale():
	""" Load any collection of images, but only their grayscale values. """

	def __init__(self, root_dir, im_size=255):
		"""
		Args:
			root_dir (string)			 : Directory with all the images of one specific class.
			im_size (int, optional)	   : Images will be padded (black) into a square of length im_size, defaults to 255.
		"""

		self.im_size = im_size

		self.root_dir = root_dir

		# get all image paths in root_dir. since you are expected to use one
		# directory for one class of data (for instance dogs) there is no need
		# to label each sample from a specific dataset.
		self.image_paths = []
		for entry in os.scandir(root_dir):
			if (entry.path.endswith(".jpeg") or entry.path.endswith(".jpg") or entry.path.endswith(".png")) and entry.is_file():
				self.image_paths += [entry.path]

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		# slicing
		if isinstance(idx, slice):
			start, stop, step = idx.indices(len(self))
			return [self[i] for i in range(start, stop, step)]
		elif isinstance(idx, int):
			# load image and convert to grayscale
			sample = Image.open(self.image_paths[idx]).convert('L')

			# apply square padding (black) if aspect ratio is not 1:1
			if sample.size[0] != self.im_size or sample.size[1] != self.im_size:
				sample.thumbnail((self.im_size, self.im_size), Image.ANTIALIAS)
				
				# edge case where thumbnail cannot guarantee exact dimensions
				if sample.size[0] != self.im_size or sample.size[1] != self.im_size:
					sample = sample.resize((self.im_size, self.im_size), Image.ANTIALIAS)
				else:	
					padded_sample = Image.new("L", (self.im_size, self.im_size))
					padded_sample.paste(sample, ((self.im_size-sample.size[0])//2, (self.im_size-sample.size[1])//2))
					sample = padded_sample

					# debugging
					# sample.show()

			return sample
		else:
			logging.critical('DogDetector.__getitem__ is not implemented for idx of type %s ' % type(idx))

		return None


class DynamicBatchDataLoader():
	def __init__(self, training_data, batch_size=1, bs_multiplier=1.001, shuffle=True):
		
		self.training_data = training_data
		
		self.shuffle = shuffle
		#if self.shuffle:
		#	random.shuffle(self.training_data)

		self.batch_size = batch_size
		self.offset = 0
		self.bs_multiplier = bs_multiplier
		self.bs_value = batch_size

	def __len__(self):
		return len(self.training_data)

	def __getitem__(self, idx):
		return self.training_data[idx]

	def __iter__(self):
		#if self.shuffle: 
		#	random.shuffle(self.training_data)

		for b in range(self.batch_size):
			x = []
			
			upper_limit = self.offset + self.batch_size
			
			if upper_limit > len(self):
				for i in range(self.offset, len(self), 1):
					x += [self.training_data[i]]
				
				upper_limit = (self.offset + self.batch_size) - len(self) 
				self.offset = 0

			#logging.debug('dynamic batch offset/upper_limit: %d/%d' % (self.offset, upper_limit))

			for i in range(self.offset, upper_limit, 1):	
				x += [self.training_data[i]]

				self.offset += 1
				if self.offset >= len(self):
					self.offset = 0
			
			yield x
	
	def add_transform(self, transf):
		self.transf = transf

	def step(self):
		""" Increase batch_size, for instance per epoch. """
		if self.bs_value * self.bs_multiplier < len(self):
			self.bs_value *= self.bs_multiplier
			self.batch_size = int(self.bs_value)

		
class DogDetector(nn.Module):
	""" Convolutional Neural Network for classification of grayscale images. """

	def __init__(self, device='cpu', im_size=32, lr=0.01, epoch=0, transf=transforms.Compose([transforms.ToTensor()]), name='classifier', clip_grad=False):
		""" 
			Args:

		"""

		super(DogDetector, self).__init__()
		
		self.im_size = im_size
		self.epoch = epoch
		self.transf = transf
		self.model_name = name

		# network topology and architecture

		# feature extractor
		self.vae = GrayVAE()
		self.vae.load('autoencoder.pth')
		self.vae.eval()
		self.vae.cpu()

		in_dim1 =  1024 
		out_dim1 = 512
		self.fc1 = nn.Linear(in_dim1, out_dim1)
		
		in_dim2 =  out_dim1 
		out_dim2 = 256
		self.fc2 = nn.Linear(in_dim2, out_dim2)
		
		in_dim3 =  out_dim2 
		out_dim3 = 128
		self.fc3 = nn.Linear(in_dim3, out_dim3)
		
		in_dim4 =  out_dim3 
		out_dim4 = 64
		self.fc4 = nn.Linear(in_dim4, out_dim4)
		
		in_dim5 =  out_dim4 
		out_dim5 = 32
		self.fc5 = nn.Linear(in_dim5, out_dim5)
		
		in_dim6 =  out_dim5 
		out_dim6 = 1
		self.fc6 = nn.Linear(in_dim6, out_dim6)

		self.LRelu = nn.LeakyReLU()
		self.BN512 = nn.BatchNorm1d(512)
		self.BN256 = nn.BatchNorm1d(256)
		self.BN128 = nn.BatchNorm1d(128)
		self.BN64 = nn.BatchNorm1d(64)
		self.BN32 = nn.BatchNorm1d(32)
		self.BN1 = nn.BatchNorm1d(1)

		self.criterion = nn.BCELoss()
		
		# read https://openreview.net/pdf?id=B1Yy1BxCZ
		self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0.00001)
		#self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=lr/100000, max_lr=lr)

		# gpu computation if possible, else cpu
		
		self.device = device
		self.cpu()

		# gradient clipping in order to prevent nan values for loss
		if clip_grad:
			for p in self.parameters():
				p.register_hook(lambda grad: torch.clamp(grad, -100, 100))




	def forward(self, x):
		x = nn.BatchNorm1d(1024)(self.vae.encode(x))

		x = self.LRelu(self.BN512(self.fc1(x)))
		x = self.LRelu(self.BN256(self.fc2(x)))
		x = self.LRelu(self.BN128(self.fc3(x)))
		x = self.LRelu(self.BN64(self.fc4(x)))
		x = self.LRelu(self.BN32(self.fc5(x)))
		x = torch.sigmoid(self.BN1(self.fc6(x)))
		return x


	def save(self, path=None):
		""" Save the DogDetector model. """

		if path:
			filename = path + '.pth'
		else:
			filename = self.model_name + '.pth'

		logging.warning('writing model into %s, do not kill this process or %s will be corrupted!' % (filename, filename))
		
		torch.save({
			'model_state_dict': self.state_dict(),
			'vae_state_dict': self.vae.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(), 
			'im_size': self.im_size,
			'epoch': self.epoch
		}, filename)
		
		logging.warning('done writing.')
		
	def load(self, path):
		checkpoint = torch.load(path)
		self.__init__(im_size=checkpoint['im_size'], epoch=checkpoint['epoch'])
		self.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

		self.vae.load_state_dict(checkpoint['vae_state_dict'])
		self.vae.eval()

	def im_transform(self, path):
		""" Open, resize and pad image at given path to self.im_size. """

		sample = Image.open(path).convert('L')

		# apply square padding (black) if aspect ratio is not 1:1
		if sample.size[0] != self.im_size or sample.size[1] != self.im_size:
			sample.thumbnail((self.im_size, self.im_size), Image.ANTIALIAS)
			
			# edge case where thumbnail cannot guarantee exact dimensions
			if sample.size[0] != self.im_size or sample.size[1] != self.im_size:
				sample = sample.resize((self.im_size, self.im_size), Image.ANTIALIAS)
			else:	
				padded_sample = Image.new("L", (self.im_size, self.im_size))
				padded_sample.paste(sample, ((self.im_size-sample.size[0])//2, (self.im_size-sample.size[1])//2))
				sample = padded_sample

		return sample

	def transform(self, path):
		""" Transform a sample input from image-path to batch-tensor. """

		# a list of paths was given
		if type(path) == list:
			batch = torch.tensor([])

			for p in path:
				# see https://discuss.pytorch.org/t/concatenate-torch-tensor-along-given-dimension/2304
				batch = torch.cat((batch, self.transf(self.im_transform(p))), 0)

			batch = batch.unsqueeze(1)
			
		# a single path was given
		else:
			# add batch_dim (1)
		 	# see https://stackoverflow.com/questions/57237352/what-does-unsqueeze-do-in-pytorch
			im = self.im_transform(path)
			batch = self.transf(im).unsqueeze(0)
			im.close()
			return batch  

		return batch

	def untransform(self, tensor):
		""" Untransform a given tensor to a PIL image and undo default normalization. """

		logging.warning('note that net.untransform only works on the default transform (net.transf)')

		return transforms.ToPILImage()(tensor)

	def fit(self, epochs, training_loader, save_per_epoch=1):
		""" Train the DogDetector for the given number of epochs on the given training dataset. 
			
		Args:
			epochs (int)								  : Number of training epochs.
			training_loader (torch.utils.data.DataLoader) : DataLoader containing the dataset.

		"""
	
		# the training loop utilizes mini-batch gradient descent
	
		continue_training = True
		while continue_training:
			for im_batch in training_loader:

				batch = torch.tensor([])
				for im in im_batch:
					batch = torch.cat((batch, self.transf(im)))

				batch = batch.unsqueeze(1)

				# outputs if the batch is left as is (batch contains only positive samples at this point)
				target = torch.tensor([[1] for b in range(batch.shape[0])]) # class 1 is a dog
				
				for b in range(len(batch)):

					# with some probability, we want to introduce negative samples, but to prevent
					# predictability based on class distribution in the training dataset, the randomness
					# is supposed to be uniformly random as well
					if random.random() < 0.5:
						if False:
							for color_channel in range(len(batch[b])):
								for row in range(len(batch[b][color_channel])):
									# black 
									batch[b][color_channel][row] = torch.tensor([-1 for j in range(self.im_size)])
						
						else:
							for color_channel in range(len(batch[b])):
								for row in range(len(batch[b][color_channel])):	
									# noise
									batch[b][color_channel][row] = torch.tensor([2*random.random()-1 for j in range(self.im_size)])

						# image changes, so output changes to class 1
						target[b] = torch.tensor([0]) # class 0 is not a dog

				batch
	
				self.loss = self.criterion(self(batch), target.float()) # note that self(batch) outputs the probability of the input being a dog, 
				                                                                     # while target holds the actual class of the input
				
				self.optimizer.zero_grad() # don't forget to zero the gradient buffers per batch !
				self.loss.backward()   # backward propagate loss
				
				self.optimizer.step()  # update the parameters
				#self.scheduler.step()  # update learning rate
				training_loader.step() # update batch size
				
				# debugging loss
				if self.epoch % 100 == 99:
					logging.info('batch loss@batch_size: %f@%d\tmini-batch: %d' % (self.loss, batch.shape[0], self.epoch))
					self.save()
				
				self.epoch += 1
				if self.epoch == epochs:
					continue_training = False
					break

def main():
	# customize logging
	log_level = logging.INFO
	logging.basicConfig(level=log_level)

	# customize your datasource here
	dogs = sys.argv[1]	 # TODO: use doc_opt instead of sys.argv
	image_size = 64		# resize and (black-border)-pad images to image_size x image_size
	data_ratio = 1		# only use the first data_ratio*100% of the dataset
	train_test_ratio = 0.6 # this would result in a train_test_ratio*100%:(100-train_test_ratio*100)% training:testing split
	batch_size = 32		 # for batch gradient descent set batch_size = int(len(data_total)*train_test_ratio*data_ratio)
	data_total = ImageGrayScale(dogs, image_size)
	#batch_size = int(len(data_total)*train_test_ratio*data_ratio)

	# split data into training:testing datasets
	training_data = data_total[:int(data_ratio*train_test_ratio*len(data_total))]
	
	# data loaders (sexy iterators)
	training_loader = DynamicBatchDataLoader(training_data, batch_size=batch_size, bs_multiplier=1.0001, shuffle=True)
	

	# customize your DogDetector here
	model_path = 'classifier.pth'
	epochs = 1000000
	learning_rate = 0.001
	save_per_epoch = 100  # save model every 100 epochs

	# create a DogDetector
	net = DogDetector(im_size=image_size, lr=learning_rate, name='classifier', device='cpu')

	# load an existing model if possible
	#net.load(model_path)

	# train the model
	net.train()
	net.fit(epochs, training_loader, save_per_epoch=save_per_epoch)

	# save/dump model
	net.save()

if __name__ == '__main__':
	main()
