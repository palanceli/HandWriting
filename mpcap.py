
# -*- coding:utf-8 -*-

import logging
import os
import numpy
import cv2
import math
import sys
import scipy
import scipy.interpolate
import unittest

class Cap(object):
	FACTOR = 1
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.img = cv2.resize(cv2.imread('dot.png'), None, fx=Cap.FACTOR, fy=Cap.FACTOR, interpolation=cv2.INTER_CUBIC)

	def getLeftTop(self):
		h, w, channels = self.img.shape
		return (self.x - w / 2, self.y - h / 2)

	def Paste2Img(self, img):
		# ToDo : 考虑点在画面外的情况
		capgray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		ret, mask = cv2.threshold(capgray, 20, 255, cv2.THRESH_BINARY)
		mask_inv = cv2.bitwise_not(mask)
		fg = cv2.bitwise_and(self.img, self.img, mask=mask)

		x, y = self.getLeftTop()
		h, w, channels = self.img.shape
		roi = img[y : y+h, x : x+w]
		bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
		dst = cv2.add(bg, fg)
		img[y : y+h, x : x+w] = dst
		return img

class CapUT(unittest.TestCase):
	def setUp(self):
	    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
	    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)

	def waitToClose(self, img):
		while True:
			cv2.imshow('image', img)
			if cv2.waitKey(20) & 0xFF == 27:
				break
		
		cv2.destroyAllWindows()

	def test01(self):
		img = numpy.zeros((300, 300, 3), numpy.uint8)
		img[:, :] = (255, 255, 255)

		cap = Cap(100, 100)
		cap.Paste2Img(img)
		self.waitToClose(img)

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main
    # cmd: python -m unittest mpcap.CapUI.Test01