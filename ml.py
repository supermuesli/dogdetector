""" universal grayscale image classifier. see https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html """

import os, logging, random, sys, math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

class ImageGrayScale():
	""" Load any collection of images, but only their grayscale values. """

	def __init__(self, root_dir, im_size=255):
		"""
		Args:
			root_dir (string)             : Directory with all the images of one specific class.
			im_size (int, optional)       : Images will be padded (black) into a square of length im_size, defaults to 255.
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
			logging.critical('CNN.__getitem__ is not implemented for idx of type %s ' % type(idx))

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

		
class CNN(nn.Module):
	""" Convolutional Neural Network for classification of grayscale images. """

	def __init__(self, device='cpu', im_size=32, lr=0.01, epoch=0, transf=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]), name='model', clip_grad=False):
		""" 
			Args:

		"""

		super(CNN, self).__init__()
		
		self.im_size = im_size
		self.epoch = epoch
		self.transf = transf
		self.name = name

		# network topology and architecture
		pool_size = 2
		self.pool = nn.MaxPool2d(pool_size, pool_size)
		
		# prefarably multiple of pool_size
		kernel_size = 4	
		padding = kernel_size//2
		stride = 1

		# <encoder>

		in_dim1 = 1   # because we only consider grayscale values (luminance)
		out_dim1 = 32
		amount_pools = 1
		self.conv1 = nn.Conv2d(in_dim1, out_dim1, kernel_size=kernel_size, padding=padding, stride=stride)
		
		in_dim2 = out_dim1
		out_dim2 = 16
		amount_pools += 1
		self.conv2 = nn.Conv2d(in_dim2, out_dim2, kernel_size=kernel_size, padding=padding, stride=stride)

		in_dim3 = out_dim2
		out_dim3 = 8
		self.conv3 = nn.Conv2d(in_dim3, out_dim3, kernel_size=kernel_size, padding=padding, stride=stride)

		# fully connected layer assuming maxpooling after every convolution.
		# we try to learn 10 principal components
		in_dim4 = 9800 #out_dim3 * (self.im_size // (pool_size**amount_pools) )**2 
		out_dim4 = 10
		self.fc1 = nn.Linear(in_dim4, out_dim4)

		# </encoder>

		# <decoder>
	
		self.fc2 = nn.Linear(out_dim4, in_dim4)

		self.deconv1 = nn.ConvTranspose2d(out_dim3, in_dim3, kernel_size=kernel_size, padding=padding, stride=stride)
		self.deconv2 = nn.ConvTranspose2d(out_dim2, in_dim2, kernel_size=kernel_size, padding=padding, stride=stride)
		self.deconv3 = nn.ConvTranspose2d(out_dim1, in_dim1, kernel_size=kernel_size, padding=padding, stride=stride)

		# </decoder>

		self.criterion = nn.MSELoss()
		
		# read https://openreview.net/pdf?id=B1Yy1BxCZ
		self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0.01)
		#self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=lr, max_lr=lr*100)

		# gpu computation if possible, else cpu
		self.device = device
		self.to(self.device)

		# gradient clipping in order to prevent nan values for loss
		if clip_grad:
			for p in self.parameters():
				p.register_hook(lambda grad: torch.clamp(grad, -100, 100))

	def forward(self, x):
		# encode
		
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		
		# flatten and keep batchsize
		shapey = x.size()
		x = x.view(x.shape[0], -1)
		
		x = self.fc1(x)
		
		# decode
		x = self.fc2(x)

		# square'en and keep batchsize
		x = x.view(x.shape[0], shapey[1], shapey[2], shapey[3])

		x = F.relu(self.deconv1(x))
		x = F.relu(self.deconv2(x))
		x = F.relu(self.deconv3(x))

		# reshape to original imagesize and keep batchsize
		x = x.view(x.size(0), 1, self.im_size, self.im_size)

		return x

	def save(self, path=None):
		""" Save the CNN model. """

		if path:
			filename = path + '.ptc'
		else:
			filename = self.name + '.ptc'

		logging.warning('writing model into %s, do not kill this process or %s will be corrupted!' % (filename, filename))
		
		torch.save({
			'model_state_dict': self.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(), 
			'im_size': self.im_size,
			'epoch': self.epoch
		}, filename)
		
		logging.warning('done writing.')
		
	def load(self, path):
		checkpoint = torch.load(path)
		self.__init__(im_size=checkpoint['im_size'], epoch=checkpoint['epoch'])
		self.load_state_dict(checkpoint['model_state_dict'])
		#self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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
			im = self.im_transform(path[0])
			batch = self.transf(im).unsqueeze(0)
			im.close()

			for i, p in enumerate(path):
				if i > 0:
					# see https://discuss.pytorch.org/t/concatenate-torch-tensor-along-given-dimension/2304
					batch = torch.cat((batch, self.transf(self.im_transform(p)).unsqueeze(0)), 0)
		
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

		return transforms.ToPILImage()((tensor+1)/2)

	def fit(self, epochs, training_loader, save_per_epoch=1):
		""" Train the CNN for the given number of epochs on the given training dataset. 
			
		Args:
			epochs (int)                                  : Number of training epochs.
			training_loader (torch.utils.data.DataLoader) : DataLoader containing the dataset.

		"""
	
		# the training loop utilizes mini-batch gradient descent
	
		continue_training = True
		while continue_training:
			for im_batch in training_loader:
				self.optimizer.zero_grad() # don't forget to zero the gradient buffers per batch !
				
				#for im in im_batch:
				#	im.show()
				#	input()

				batch = torch.tensor([])
				for im in im_batch:
					batch = torch.cat((batch, self.transf(im).unsqueeze(0)))
					
				self.loss = self.criterion(self(batch), batch) 

				
				# debugging loss
				if self.epoch % 10 == 9:
					logging.info('batch loss@batch_size: %f@%d\tepoch: %d' % (self.loss, batch.shape[0], self.epoch))
				
				self.loss.backward()   # backward propagate loss
				
				self.optimizer.step()  # update the parameters
				#self.scheduler.step()  # update learning rate
				training_loader.step() # update batch size
				
				# epoch is finished at this point
				if self.epoch % save_per_epoch == save_per_epoch - 1:
					self.save()

				self.epoch += 1
				if self.epoch == epochs:
					continue_training = False
					break

def main():
	# customize logging
	log_level = logging.INFO
	logging.basicConfig(level=log_level)

	# if CUDA available, use it
	if torch.cuda.is_available():  
		dev = 'cuda:0' 
	else:  
		dev = 'cpu'  
	device = torch.device(dev) 

	# customize your datasource here
	dogs = sys.argv[1]     # TODO: use doc_opt instead of sys.argv
	image_size = 32        # resize and (black-border)-pad images to image_size x image_size
	data_ratio = 0.1         # only use the first data_ratio*100% of the dataset
	train_test_ratio = 0.6 # this would result in a train_test_ratio*100%:(100-train_test_ratio*100)% training:testing split
	batch_size = 512         # for batch gradient descent set batch_size = int(len(data_total)*train_test_ratio*data_ratio)
	data_total = ImageGrayScale(dogs, image_size)
	#batch_size = int(len(data_total)*train_test_ratio*data_ratio)

	# split data into training:testing datasets
	training_data = data_total[:int(data_ratio*train_test_ratio*len(data_total))]
	
	# data loaders (sexy iterators)
	training_loader = DynamicBatchDataLoader(training_data, batch_size=batch_size, bs_multiplier=1, shuffle=True)
	

	# customize your CNN here
	model_path = 'autoencoder.ptc'
	epochs = 1000000
	learning_rate = 0.1
	save_per_epoch = 100  # save model every 100 epochs

	# create a CNN
	net = CNN(im_size=image_size, lr=learning_rate, name='autoencoder')

	# load an existing model if possible
	#net.load(model_path)

	# train the model
	net.train()
	net.fit(epochs, training_loader, save_per_epoch=save_per_epoch)

	# save/dump model
	net.save()

if __name__ == '__main__':
	main()
