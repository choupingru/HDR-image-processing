import numpy as np
import os
from PIL import Image
import cv2
from os.path import join
from os import listdir
import random
import matplotlib.pyplot as plt
import argparse
from utils import ToneMapping, gSolver, ImageAlignment

parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
parser.add_argument('-dir', '--dir', default='jpgfile_1', type=str)
args = parser.parse_args()

USE_ALIGNMENT = True

if __name__ == '__main__':
	image_path = join(os.getcwd(), args.dir)
	files = image_path

	alignment_tool = ImageAlignment()
	if USE_ALIGNMENT:
		files = sorted([file for file in listdir(image_path) if not file.startswith('.')])
		src_index = len(files) // 2
		src_image = None
		for index, tar in enumerate(files):
			if index != src_index:
				alignment_tool.setSrcTar(join(image_path, files[src_index]), join(image_path, tar))
				src, tar = alignment_tool.solve()
				files[index] = tar
				if src_image is None:
					src_image = src
		files[src_index] = src_image
		files = np.array(files)

	g_solver = gSolver( 
		image_path=files, \
		exposure_times=np.array([1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2], 'float32'),
	)

	hdr = g_solver.fit(name="align" if USE_ALIGNMENT else "no_align")

	tone_mapping = ToneMapping()
	output = tone_mapping.fast_bilateral_filter(hdr)
	cv2.imwrite("./output/tonemapping_output.jpg",output)
	