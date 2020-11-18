import torch

def conv2d(ten, ind, outd, ks=1, padding=1):
	
	x = torch.tensor([ [ [[0 for y in range(ten.shape[3])] for x in range(ten.shape[2])] for o in range(outd)] for batch in range(ten.shape[0])] )

	for batch in range(ten.shape[0]):
		for o in range(outd):
			for i in range(ind):
				for x in range(ten.shape[2]):
					for y in range(ten.shape[3]):
						# convolve and concatenate
						conv_val = ten[batch][i][x][y]

						x[b][o]
						

	return x

def main():
	a = torch.tensor([ [[ [1,2,3], [4,5,6] , [7,8,9]]  ] ,\
	                   [[ [5,6,7], [8,9,10], [11,12,13]] ] ]) # 2 batches, 1 channel

	print(a.shape, a)

	b = conv2d(a, 1, 3, 2)

	print(b.shape)
	print(b)

if __name__ == '__main__':
	main()
