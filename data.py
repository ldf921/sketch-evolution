from scipy.misc import imread, imresize
import os 
import re
import random
from tqdm import tqdm
import numpy as np

class FileDataSet:
	def __init__(self, path, photo, sketch):
		files = []
		for f in os.listdir(path):
			if re.match(photo, f):
				files.append(f)

		files = list(sorted(files))
		
		self.imgs = []
		for f in files:			
			sketch_f = sketch % re.match(photo, f).group(1)
			photo_img = imread(os.path.join(path, f))
			sketch_img = imread(os.path.join(path, sketch_f))
			self.imgs.append( [photo_img, sketch_img] ) 

def get_data(name):
	if name == 'bird':
		return FileDataSet('../birds', r'^(\w+\d+).jpg', '%s_FotoSketcher.jpg')
	elif name == 'butterfly':
		return FileDataSet('../butterflies', r'^(\w+\d+).jpg', '%s_FotoSketcher.jpg')
	elif name == "lfw":
		return FileDataSet('../lfw', r'^(\w+\d+).jpg', '%s_FotoSketcher.jpg')
	else:
		raise ValueError("Invalid data set {}".format(name))

class DataProvider:
	def __init__(self, imgs, **kwargs):
		self.imgs = imgs
		self.length = len(imgs)

	def sample(self, batch_size, sketch_shape, output_shape, groundtruth = False):
		''' return batch_size images
		'''

		sketch_image = []
		truth_image = []
		other_image = []

		batchimg = random.sample(self.imgs, batch_size)
		for img in batchimg:
			sketch_image.append(imresize(img[1], sketch_shape) )
			truth_image.append(imresize(img[0], output_shape) )

		batchimg = random.sample(self.imgs, batch_size)
		for img in batchimg:
			other_image.append(imresize(img[0], output_shape) )

		ret = dict()
		ret['sketch_image'] = sketch_image
		if groundtruth:
			ret['truth_image'] = truth_image
		ret['other_image'] = other_image
		return ret

class DataProviderSimple:
	def __init__(self, imgs, sketch_shape, output_shape, **kwargs):
		self.length = len(imgs)

		aug_imgs = []

		for img in tqdm(imgs):

			img[1] = imresize(img[1], sketch_shape)
			img[0] = imresize(img[0], output_shape)

			flag = True
			for x in range(2):
				if len(img[x].shape) != 3:
					flag = False
					# print(img[x].shape)
					# img[x] = np.repeat( np.expand_dims(img[x], 2), 3, axis = 2)
					# print(img[x])
					# break

			if flag:
				current_imgs = [img]
				for img in current_imgs:
					aug_imgs.append(img)
					aug_imgs.append(tuple(map(np.fliplr, img)))

		self.imgs = aug_imgs
		

	def sample(self, batch_size, sketch_shape, output_shape, groundtruth = False, **kwargs):
		''' return batch_size images
		'''

		sketch_image = []
		truth_image = []
		other_image = []

		batchimg = random.sample(self.imgs, batch_size)
		for img in batchimg:
			sketch_image.append(img[1] )
			truth_image.append(img[0] )

		batchimg = random.sample(self.imgs, batch_size)
		for img in batchimg:
			other_image.append(img[0] )

		ret = dict()
		ret['sketch_image'] = sketch_image
		if groundtruth:
			ret['truth_image'] = truth_image
		ret['other_image'] = other_image
		return ret

if __name__ == '__main__':
	ds = get_data('lfw')
	print(len(ds.imgs))