""" universal grayscale image classifier. see https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html """

import os, logging, random, sys

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
			transform (callable, optional): Transform to be applied on a sample, defaults to transforms.ToTensor().
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

			return transforms.ToTensor()(sample)
		else:
			logging.critical('CNN.__getitem__ is not implemented for idx of type %s ' % type(idx))

		return None


class DynamicBatchDataLoader():
	def __init__(self, training_data, batch_size=1, bs_multiplier=1.001, shuffle=True):
		
		self.training_data = training_data
		
		self.shuffle = shuffle
		if self.shuffle:
			random.shuffle(self.training_data)

		self.batch_size = batch_size
		self.offset = 0
		self.bs_multiplier = bs_multiplier
		self.bs_value = batch_size

	def __len__(self):
		return len(self.training_data)

	def __getitem__(self, idx):
		return self.training_data[idx]

	def __iter__(self):
		if self.shuffle: 
			random.shuffle(self.training_data)

		for b in range(self.batch_size):
			x = torch.tensor([])
			
			upper_limit = self.offset + self.batch_size
			
			if upper_limit > len(self):
				for i in range(self.offset, len(self), 1):	
					x = torch.cat((x, self.training_data[i].unsqueeze(0)), 0)
				
				upper_limit = (self.offset + self.batch_size) - len(self) 
				self.offset = 0

			#logging.debug('dynamic batch offset/upper_limit: %d/%d' % (self.offset, upper_limit))

			for i in range(self.offset, upper_limit, 1):	
				x = torch.cat((x, self.training_data[i].unsqueeze(0)), 0)

				self.offset += 1
				if self.offset >= len(self):
					self.offset = 0
			
			yield x

	def step(self):
		""" Increase batch_size, for instance per epoch. """
		if self.bs_value * self.bs_multiplier < len(self):
			self.bs_value *= self.bs_multiplier
			self.batch_size = int(self.bs_value)

		
class CNN(nn.Module):
	""" Convolutional Neural Network for classification of grayscale images. """

	def __init__(self, device='cpu', im_size=100, lr=0.01, transf=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])):
		super(CNN, self).__init__()
		
		self.im_size = im_size
		self.transf = transf

		# network topology (encoder)
		kernel_size = 5	
		pool_size = 2
		out_dim = 6
		self.conv1 = nn.Conv2d(1, out_dim, kernel_size)
		self.pool = nn.MaxPool2d(pool_size, pool_size)
		
		in_dim = out_dim
		out_dim = 16
		self.conv2 = nn.Conv2d(in_dim, out_dim, kernel_size)
		self.fc1 = nn.Linear(out_dim * ((((im_size - kernel_size) // pool_size) - kernel_size) // pool_size)**2, 64) # consider self.forward as to why this input dimension was chosen, also read here:
		                                                                                                             # https://stackoverflow.com/questions/53784998/how-are-the-pytorch-dimensions-for-linear-layers-calculated
		self.fc2 = nn.Linear(64, 32)
		self.fc3 = nn.Linear(32, 16)
		self.fc4 = nn.Linear(16, 8)
		
		# decoder
		self.fc5 = nn.Linear(8, 16)
		self.fc6 = nn.Linear(16, 32)
		self.fc7 = nn.Linear(32, 64)
		self.fc8 = nn.Linear(64, im_size*im_size)

		#self.criterion = nn.MSELoss()
		self.criterion = lambda x, y: ((y-x)**2).mean()

		# read https://openreview.net/pdf?id=B1Yy1BxCZ
		self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
		self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=lr, max_lr=lr*100)

		# gpu computation if possible, else cpu
		self.device = device
		self.to(self.device)

		# gradient clipping in order to prevent nan values for loss
		#for p in self.parameters():
		#	p.register_hook(lambda grad: torch.clamp(grad, -100, 100))

	def forward(self, x):

		# extract features (encode)
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, self.fc1.in_features) # flatten the self.conv2 convolution layer. 
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		x = self.fc3(x)
		x = self.fc4(x)

		# decode
		x = self.fc5(x)
		x = self.fc6(x)
		x = self.fc7(x)
		x = self.fc8(x)

		x = x.view(-1, self.im_size, self.im_size) # squeeze output into batch_dim
		x = x.unsqueeze(1)

		#with torch.no_grad():
		#	im = self.untransform(x[0])
		#	im.show()
		#	input('press enter to continue')

		return x

	def save(self, path):
		""" Save the CNN model as an altered state dict (.asd). """

		# manually add self.im_size to state_dict so we can restore it when loading the model again
		sd = self.state_dict()
		sd['im_size'] = self.im_size
	
		logging.warning('writing model into %s.asd, do not kill this process or %s.asd will be corrupted!' % (path, path))
		torch.save(sd, path + '.asd')
		logging.warning('done writing.')
		
	def load(self, path):
		sd = torch.load(path)
		
		# load non-conventional attributes from altered state dict (if they exist)
		try:
			self.im_size = sd['im_size']
			del sd['im_size']
		
			self.__init__(im_size=self.im_size)
		except KeyError as e:
			logging.critical(e)
		# load conventional state dict
		
		self.load_state_dict(sd)

	def im_transform(self, path):
		""" Resize and pad image to self.im_size. """

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
		""" Transform a sample input so it fits through the network topology. """

		# a list of paths was given
		if type(path) == list:
			batch = self.transf(self.im_transform(path[0])).unsqueeze(0)
			for i, p in enumerate(path):
				if i > 0:
					# see https://discuss.pytorch.org/t/concatenate-torch-tensor-along-given-dimension/2304
					batch = torch.cat((batch, self.transf(self.im_transform(p)).unsqueeze(0)), 0)
		# a single path was given
		else:
			# add batch_dim (1)
		 	# see https://stackoverflow.com/questions/57237352/what-does-unsqueeze-do-in-pytorch
			return self.transf(self.im_transform(path)).unsqueeze(0)  

		return batch


	def untransform(self, tensor):

		logging.warning('note that net.untransform only works on the default transform (net.transf)')

		return transforms.ToPILImage()((tensor+1)/2)

	def fit(self, cycles, training_loader, save_per_cycle=1):
		""" Train the CNN for the given number of cycles on the given training dataset. 
			
		Args:
			cycles (int)                                  : Number of training cycles.
			training_loader (torch.utils.data.DataLoader) : DataLoader containing the dataset.

		"""
	
		# the training loop utilizes mini-batch gradient descent

		cycle = 0		
		continue_training = True
		while continue_training:
			for batch in training_loader:
				self.optimizer.zero_grad() # don't forget to zero the gradient buffers per batch !
				
				#for b in batch:
				#	im = self.untransform(b)
				#	im.show()
				#	input()

				self.loss = self.criterion(self(batch), batch) 
				
				# debugging loss
				if cycle % 10 == 9:
					logging.info('batch loss@batch_size: %f@%d\tcycle: %d' % (self.loss, batch.shape[0], cycle))
				
				self.loss.backward()   # backward propagate loss
				self.optimizer.step()  # update the parameters
				#self.scheduler.step()  # dynamic learning rate
				training_loader.step() # dynamic batch size
				
				# cycle is finished at this point
				if cycle % save_per_cycle == save_per_cycle - 1:
					self.save('model')

				cycle += 1
				if cycle == cycles:
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
	image_size = 115        # resize and (black-border)-pad images to image_size x image_size
	data_ratio = 0.01         # only use the first data_ratio*100% of the dataset
	train_test_ratio = 0.6 # this would result in a train_test_ratio*100%:(100-train_test_ratio*100)% training:testing split
	batch_size = 64         # for batch gradient descent set batch_size = int(len(data_total)*train_test_ratio*data_ratio)
	data_total = ImageGrayScale(dogs, image_size)
	#batch_size = int(len(data_total)*train_test_ratio*data_ratio)

	# split data into training:testing datasets
	training_data = data_total[:int(data_ratio*train_test_ratio*len(data_total))]
	
	# data loaders (sexy iterators)
	training_loader = DynamicBatchDataLoader(training_data, batch_size=batch_size, bs_multiplier=1.001, shuffle=True)
	

	# customize your CNN here
	model_path = 'model.asd'
	cycles = 1000000
	learning_rate = 0.0000000000000001
	save_per_cycle = 100  # save model every 100 cycles
	transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

	# create a CNN
	net = CNN(im_size=image_size, lr=learning_rate, transf=transf)

	# load an existing model if possible
	#net.load(model_path)

	# train the model
	net.fit(cycles, training_loader, save_per_cycle=save_per_cycle)

	# save/dump model
	net.save("model")

if __name__ == '__main__':
	main()
