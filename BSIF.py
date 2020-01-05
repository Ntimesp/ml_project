import numpy as np 
import math
import os
import cv2
import scipy.linalg as sl
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from tqdm import tqdm

def whiten(sample, n): #input samples as an r*m matrix, n represents the number of eigenvalues we select#
	cov = np.cov(sample)
	eig_vec, eig_val, temp= np.linalg.svd(cov)
	diag_val = [math.sqrt(1/k) for k in eig_val]
	d = np.diag(diag_val)
	wt_mat = np.dot(d, np.transpose(eig_vec))
	wt_mat = wt_mat[0:n, :]
	return wt_mat
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
def cosh(x):
	return (np.exp(x) + np.exp(-x)) / 2
def cosh2(x):
	return pow(cosh(x), -2)
def g(x):
	return x * math.exp(- math.pow(x,2.0) /2.0)
def gd(x):
	return (1.0 - math.pow(x, 2.0) ) * math.exp(-math.pow(x,2.0)/2.0)
def ica(data): #input data as an n*m matrix#
	n, m = len(data[:, 0]), len(data[0, :])
	W = np.diag(np.ones(n))
	sam = np.zeros((n, m))
	W_old = np.zeros((n, n))
	
	itera = 0
	while(1):
		itera += 1
		if(itera >= 10000):
			return 'error'
		W_old = W
		for i in range(0, n):
			for j in range(0, m):
				sam[:, j] = data[:, j] * tanh((np.dot(np.transpose(W_old[:, i]), data[:, j]))) - W_old[:, i]*cosh2((np.dot(np.transpose(W_old[:, i]), data[:, j])))
			W[:, i] = np.mean(sam, axis = 1)
			if(i >= 1):
				temp = np.zeros((1, n))
				for j in range(0, i):
					temp = temp + np.dot(np.transpose(W[:, i]), W[:, j]) * W[:, j]
				W[:, i] = W[:, i] - temp
			W[:, i] = W[:, i] / np.linalg.norm(W[:, i])
		if (np.linalg.norm(W - W_old) <= 1e-25):
			return W

def sampling(dir, scale):#dir denotes the situation of your files, and scale denotes your scale of samples#
	for parents, dirnames, filenames in os.walk(dir):
		num = len(filenames)
	samples = np.zeros((scale * scale, 50 * 50 * num))
	i = 0
	for parents, dirnames, filenames in os.walk(dir):
		for filename in filenames:
			img = cv2.imread(os.path.join(dir, filename), cv2.IMREAD_GRAYSCALE)
			for k in range(0, int(len(img[:, 0])/scale)):
				for l in range(0, int(len(img[0, :])/scale)):
					samples[:, i] = np.ravel(img[k * scale: (k+1) * scale, l * scale: (l+1) * scale])
					i = i + 1
	return samples
def train_filters(dir):
	for parents, dirnames, filenames in os.walk(dir):
		filename = filenames[0]
	img = cv2.imread(os.path.join(dir, filename), cv2.IMREAD_GRAYSCALE)
	scale_img = [len(img[:, 0]), len(img[0, :])]
	ss = [3,5,7,9,11,13,15,17]
	res = np.zeros((scale_img[0], scale_img[1], 8))
	filterss = []
	for k in tqdm(ss):
		samples = sampling(dir, k)
		v = whiten(samples, 8)
		wt = np.dot(v, samples)
		u = ica(wt)
		f = np.dot(u,v)
		filters = np.zeros((k, k, 8))
		for i in range(0, 8):
			filters[:, :, i] = f[i, :].reshape((k, k),order = 'C')
		filterss.append(filters)
		np.save("filters.npy", filterss)
	return filterss

def BSIF(img, x = 10, y = 10): 
#x denotes the number of rows in each partition, while y denotes the number of columns#
#返回一个2维array，每列代表图片某个分块的柱状图#
	filterss = np.load ("filters.npy", allow_pickle=True)
	scale_img = [len(img[:, 0]), len(img[0, :])]
	num_row, num_column = int(scale_img[0]/x), int(scale_img[1]/y)
	res = np.zeros((scale_img[0], scale_img[1], 8))
	flag = 0
	for filters in filterss:
		res[:, :, flag] = show_filter(filters, img, 8)
		flag = flag + 1
	result = np.zeros((8*256, num_row*num_column))
	flagg = 0
	for i in range(1, num_row+1):
		for j in range(1, num_column+1):
			for l in range(1, 9):
				for xx in res[(i-1)*x:i*x, (j-1)*y:j*y, l-1]:
					for yy in xx:
						result[(l-1)*256+int(yy)-1, (i-1)*num_column+j-1] = result[(l-1)*256+int(yy)-1, (i-1)*num_column+j-1] + 1
	return result
	
def show_filter(filters, eg, n):#n represents the scale of filters
	k = len(eg[:, 0])
	l = len(eg[0, :])
	after = np.zeros((k, l, n))
	img_after = np.zeros((k, l))
	for i in range(0, n):
		after[:, :, i] = convolve2d(eg, filters[:, :, i], mode = 'same', boundary = 'wrap')
		for ki in range(0, k):
			for li in range(0, l):
				if (after[ki, li, i] > 0):
					img_after[ki, li] = img_after[ki, li] + math.pow(2, i)
	return(img_after)