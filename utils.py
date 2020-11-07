import numpy as np
import cv2
from PIL import Image
from os import listdir
import cv2
import os
from os.path import join
import random
import matplotlib.pyplot as plt

class ImageAlignment(object):

	def __init__(self, depth=8, bias=4):
		
		self.bias = bias
		self.depth = depth

	def setSrcTar(self, src, tar):
		self.src = self.readImage(src)
		self.tar = self.readImage(tar)

	def readImage(self, path):
		image = cv2.imread(path)
		return image

	def shiftImage(self, input, dx, dy):
		# dy --> up down
		# dx --> left right
		w, h = input.shape[:2]
		T = np.array(
			[[1, 0, dx],
			 [0, 1, dy]]
		, dtype='float32')
		image_translation = cv2.warpAffine(input, T, (h, w))
		return image_translation

	def shirnkImage(self, input, ratio=0.5):
		w, h = input.shape[:2]
		w, h = int(w * ratio), int(h * ratio)
		input = cv2.resize(input, dsize=(h, w))
		return input

	def rgb2gray(self, rgb):
		rgb = rgb.astype(np.int64)
		r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
		grey = (54 * r + 183 * g + 19 * b) // 256
		return grey.astype(np.uint8)
		

	def medianThreshold(self, input):
		median = np.median(input)
		image = input.copy()
		image = 1 * (image > median)
		return image

	def exclusionMap(self, input, bias=4):
		median = np.median(input)
		image = input.copy()
		exclusion = 1 * (image > median + bias) + 1 * (image < median - bias)
		return exclusion

	def xorDifference(self, src, tar):
		diff = np.logical_xor(src, tar)
		return diff

	def fit(self, src, tar, prev_dx, prev_dy):
		h, w = src.shape[:2]
		best_error, best_dx, best_dy = float('inf'), 0, 0
		src_median = self.medianThreshold(src)
		src_exclude = self.exclusionMap(src)
		for dx in range(-1, 2):
			for dy in range(-1, 2):
				tar_shift = self.shiftImage(tar, dx, dy)
				tar_median = self.medianThreshold(tar_shift)
				tar_exclude = self.exclusionMap(tar_shift)
				error = np.sum(self.xorDifference(src_median, tar_median) * src_exclude * tar_exclude)
				if error < best_error:
					best_dx = dx
					best_dy = dy
					best_error = error
		
		return prev_dx + best_dx, prev_dy + best_dy
		
	def align(self, src, tar, depth):

		if depth == 0:
			dx, dy = self.fit(src, tar, 0, 0)
			return dx, dy
		else:
			nextLevel_src = self.shirnkImage(src, 0.5)
			nextLevel_tar = self.shirnkImage(tar, 0.5)
			prev_dx, prev_dy = self.align(nextLevel_src, nextLevel_tar, depth-1)
			tar_shift = self.shiftImage(tar, prev_dx * 2, prev_dy * 2)
			dx, dy = self.fit(src, tar_shift, prev_dx * 2, prev_dy * 2)
			return dx, dy

	def solve(self):
		src_gray = self.rgb2gray(self.src)
		tar_gray = self.rgb2gray(self.tar)
		dx, dy = self.align(src_gray, tar_gray, self.depth)
		print('X offset : {}, Y offset : {}'.format(dx, dy))
		self.tar = self.shiftImage(self.tar, dx, dy)
		return self.src, self.tar



class gSolver(object):

	def __init__(self, image_path, exposure_times, sohw_radiance_map=True, show_g_function=True):
		
		if isinstance(image_path, str):
			files = sorted(listdir(image_path))
			self.images = [np.array(cv2.imread(join(image_path, file))) for file in files if not file.startswith('.')]
			self.images = np.array(self.images)
		elif isinstance(image_path, np.ndarray):
			self.images = image_path

		self.sample_number = 3000 // (self.images.shape[0] - 1)
		self.B = exposure_times
		
		self.show_g_function = show_g_function
		self.sohw_radiance_map = sohw_radiance_map

	def weight(self, val):
		return abs(128 - val)

	def solver(self, Z, B, l=10):
		
		B = np.log(B)
		n, P = Z.shape # samples, images
		A = np.zeros(shape=(n * P + 255, 256 + n))
		b = np.zeros(A.shape[0])

		k = np.array([i for i in range(n * P)])
		pixel_val = Z.reshape(-1)
		w_lnE = np.array([256 + i for i in range(n) for j in range(P)])
		b_index = np.array([j for i in range(n) for j in range(P)])
		weights = self.weight(pixel_val)


		A[k, pixel_val] = weights
		A[k, w_lnE] = weights * -1
		b[k] = B[b_index] * weights
		A[n*P][128] = 1

		k = np.array([n*P + i for i in range(1, 255)])

		weights = np.array([[self.weight(i), -2 * self.weight(i), self.weight(i)] for i in range(1, 255)])
		for index, k_ in enumerate(k):
			A[k_][index+1:index+4] = weights[index] * l

		x, _, _, _ = np.linalg.lstsq(A, b, rcond=-1)

		g = x[:256]
		lnE = x[256:]
		return g, lnE


	def hdr_con(self, g, I, ln_te, weight_fun):
		lnE = np.zeros(I[0].shape)
		ln_te = np.log2(ln_te)
		ln_te_mat = np.array([np.tile(ln_te[i], I.shape[1:-1]) for i in range(I.shape[0])])
		for ch in range(3):
			weighted_sum = np.sum(weight_fun(I[:,:,:,ch]) * (g[ch][I[:,:,:,ch]] - ln_te_mat), axis=0)
			weight_sum = np.sum(weight_fun(I[:,:,:,ch]), axis=0)
			lnE[:,:,ch] = weighted_sum / weight_sum
		return lnE

	def fit(self, name=""):
		# sample from image
		# B = self.

		row, col = self.images[0].shape[:2]
		candidate = []
		for sample in range(self.sample_number):
			candidate += [(random.randint(1, row-1),  random.randint(1, col-1))]

		# get the sample pixel from image
		Z = [[] for i in range(len(candidate))] # (100, 1)
		for index, (row, col) in enumerate(candidate):
			for img in self.images:
				Z[index].append(img[row][col]) # (100, num_images)
		Z = np.array(Z)
		r_g, r_lnE = self.solver(Z[:,:,0], self.B, 10)
		g_g, g_lnE = self.solver(Z[:,:,1], self.B, 10)
		b_g, b_lnE = self.solver(Z[:,:,2], self.B, 10)


		g = np.array([b_g, g_g, r_g])
		hdr = self.hdr_con(g, self.images, self.B, self.weight)
		hdr = np.exp(hdr)	

		cv2.imwrite("./output/{}_output.hdr".format(name),hdr)

		if self.show_g_function:
			colors = ['r', 'g', 'b']
			for index, channel in enumerate(g):
				xs, ys = [], []
				for pixel_value, lnE in enumerate(channel):
					xs += [lnE]
					ys += [pixel_value]
				plt.scatter(xs, ys, color=colors[index], s=0.1)
			plt.savefig('./output/{}_g_functino.png'.format(name))

		if self.sohw_radiance_map:
			plt.figure(figsize=(12, 8))
			plt.imshow(np.log2(cv2.cvtColor(hdr.astype(np.float32), cv2.COLOR_BGR2GRAY)), cmap='jet')
			plt.colorbar()
			plt.savefig('./output/{}_radiance_map.png'.format(name))

		return hdr


class ToneMapping(object):

	def __init__(self):
		pass

	def fast_bilateral_filter(self, hdr, mode='rgb'):
		
		hdr = hdr.astype(np.float32)

		w, h = hdr.shape[:2]

		rw, gw, bw = 20, 40, 1
		r, g, b = hdr[..., 0], hdr[..., 1],  hdr[..., 2]
		if mode == 'bgr':
			r, b = b, r
		intensity = rw * hdr[..., 0] + gw * hdr[..., 1] + bw * hdr[..., 2]
		r, g, b = r / (intensity + 1e-5), g / (intensity + 1e-5), b / (intensity + 1e-5)

		log_base = cv2.bilateralFilter(np.log2(intensity), 3, 15, 15)
		log_detail = np.log2(intensity) - log_base
		compression_factor = 6 / (np.max(log_base) - np.min(log_base))
		log_output_intensity = compression_factor * (log_base - np.max(log_base)) + log_detail

		intensity_weights = np.power(2, log_output_intensity)

		r_out, g_out, b_out = r * intensity_weights, g * intensity_weights, b * intensity_weights

		output = np.zeros(hdr.shape)
		output[:, :, 0] = r_out
		output[:, :, 1] = g_out
		output[:, :, 2] = b_out

		output = np.clip((output ** (1/2.2)), 0, 255)
		output = (output - output.min()) / (output.max() - output.min()) * 255
		return output.astype(np.uint8)


if __name__ == '__main__':
	pass
	### Tonemapping example
	# hdr = cv2.imread('./output/no_align_output.hdr')
	# tool = ToneMapping()
	# output = tool.fast_bilateral_filter(hdr)
	# cv2.imwrite("./output/tonemapping_output.jpg",output)
	

	### Alignment example
	# x = ImageAlignment()
	# x.setSrcTar('./test1.JPG', './test1.JPG')
	# src, tar = x.solve()