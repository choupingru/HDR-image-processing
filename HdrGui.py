from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.config import Config
from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty
import os
from os.path import join
import numpy as np
import cv2
from PIL import Image
import random
import matplotlib.pyplot as plt
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
Config.set('graphics', 'width', '1500')
Config.set('graphics', 'height', '1200')
ROOT = os.getcwd()


def weight(val):
	return abs(val - 128)

def solve(Z, B, l):
		B = np.log(B)
		n, p = Z.shape
		A = np.zeros(shape=(n*p + 256 - 1, 256+n))
		b = np.zeros(A.shape[0])
		k = 0

		for n_index, vals in enumerate(Z):
			for p_idx, val in enumerate(vals):
				val = int(val)
				A[k][val] = weight(val)
				A[k][256+n_index] = -1*weight(val)
				b[k] = B[p_idx]*weight(val) 
				k += 1
		
		A[k][128] = 1
		k += 1
		
		for i in range(1, 255):
			A[k][i] = 1*l*weight(i)
			A[k][i+1] = -2*l*weight(i)
			A[k][i+2] = 1*l*weight(i)
			k+=1
		
		x, _, _, _ = np.linalg.lstsq(A, b,rcond=-1)
		g = x[:256]
		lnE = x[256:]
		
		return g, lnE

def hdr_con(g, I, ln_te, weight_fun = weight):
	lnE = np.zeros(I[0].shape)
	ln_te = np.log(ln_te)
	ln_te_mat = np.array([np.tile(ln_te[i], I.shape[1:-1]) for i in range(I.shape[0])])

	for ch in range(3):
		weighted_sum = np.sum(weight_fun(I[:,:,:,ch]) * (g[ch][I[:,:,:,ch]] - ln_te_mat), axis=0)
		weight_sum = np.sum(weight_fun(I[:,:,:,ch]), axis=0)
		lnE[:,:,ch] = weighted_sum / weight_sum
	return lnE


def defaultWeight(val):
	if val > 128:
		return 255 - val
	else:
		return val - 0


class LoadDialog(FloatLayout):
	load = ObjectProperty(None)
	cancel = ObjectProperty(None)
	img_id = ObjectProperty(None)
	def getPath(self):
		return join(os.getcwd(),"Data")


class HomePage(Screen):
	pass

class Start(Screen):
	def dismiss_popup(self):
		self._popup.dismiss()

	def show_load(self, img_id):

		content = LoadDialog(load=self.load, cancel=self.dismiss_popup, img_id=img_id)
		self._popup = Popup(title="Load file", content=content,
		size_hint=(0.7, 0.7))
		self._popup.open()

	def load(self, img_id, path, filename):
		filename = filename[-1]
		filename = str(filename)
		self.ids[img_id].source = str(filename)
		# with open(ROOT) as stream:
			# self.text_input.text = stream.read()
		self.dismiss_popup()


	
	def select(self, filename):
		self.ids.image.source=filename[0]

	def getG(self, imgName, expos, smooth):
		# delete unvalid data
		expos = [ele for ele in expos if len(ele) < 10]
		imgName = [ele for ele in imgName if len(ele) > 0]
		# change img name to np array
		imgName = [file.split('/')[-1] for file in imgName]
		images = [Image.open(join('Data',filename)) for filename in imgName]
		images = np.array([np.array(img) for img in images])
		# change expose time to float number
		
		expos = [float(exp.split('/')[0]) if len(exp.split('/')) == 1 else float(exp.split('/')[0])/float(exp.split('/')[1]) for exp in expos]
		try:
			smooth = float(smooth)
		except:
			smooth = 1
		# size of image
		size = images[0].shape
		
		candidate = [(random.randint(1,size[0]-1), random.randint(1, size[1]-1)) for _ in range(50)]
		Z = np.array([[img[ele[0]][ele[1]] for img in images] for ele in candidate])
		# create exposure time matrix
		B = expos
		# solve the g function
		xs, ys = [],[]
		r_g, r_lnE = solve(Z[:,:,0], B, 3)
		g_g, g_lnE = solve(Z[:,:,1], B, 3)
		b_g, b_lnE = solve(Z[:,:,2], B, 3)

		g = np.array([r_g, g_g, b_g])
		hdr = hdr_con(g, images, B)
		hdr = np.exp(hdr)	
		
		for index, ele in enumerate(r_g):
			print(index)
			xs.append(ele)
			ys.append(index)
		
		fig = plt.figure()
		fig.add_subplot(1, 2, 1)
		for x, y in zip(xs, ys):
			plt.scatter(x,y)
		try:
			self.manager.screens[3].ids.gfunction.remove_widget(self.ids.gfunction.children[0])
		except:
			pass
		try:
			self.manager.screens[3].ids.gfunction.remove_widget(self.ids.gfunction.children[1])
		except:
			pass
		fig.add_subplot(1, 2, 2)
		plt.imshow(hdr)
		fig = FigureCanvasKivyAgg(fig, id="figure")
		self.manager.screens[3].ids.gfunction.add_widget(fig)
		
		
	pass

class About(Screen):
	pass

class Ggraph(Screen):
	pass



class HDR(App):
	def build(self):
		sm = ScreenManager()
		sm.add_widget(HomePage(name='homepage'))
		sm.add_widget(Start(name='start'))
		sm.add_widget(About(name='about'))
		sm.add_widget(Ggraph(name='ggraph'))
		return sm

HDR().run()

