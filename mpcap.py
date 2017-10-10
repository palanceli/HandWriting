
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

def WaitToClose(img):
	while True:
		cv2.imshow('image', img)
		if cv2.waitKey(20) & 0xFF == 27:
			break
	
	cv2.destroyAllWindows()

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

class CapTanPointHelper(object):
	def __init__(self):
		self.capX, self.capY = 25, 25
		self.cap = Cap(self.capX, self.capY)

		self.bgWidth, self.bgHeight = 50, 50
		self.bgImg = numpy.zeros((self.bgWidth, self.bgHeight, 3), numpy.uint8)
		self.bgImg[:, :] = (0, 0, 0)

		self.lineColor = 31

		self.radius = 40

	def getArmLine(self, angle, step):
		line = {'startPt': (
			int(self.cap.x - self.radius * math.cos(math.radians(angle)) + step * math.sin(math.radians(angle))),
			int(self.cap.y - self.radius * math.sin(math.radians(angle)) - step * math.cos(math.radians(angle)))) ,
			'endPt': (
			int(self.cap.x + self.radius * math.cos(math.radians(angle)) + step * math.sin(math.radians(angle))),
			int(self.cap.y + self.radius * math.sin(math.radians(angle)) - step * math.cos(math.radians(angle))))}
		return line

	def addArmLineToCap(self, armLine):
		capImg = self.bgImg.copy()
		self.cap.Paste2Img(capImg)
		bgImg = self.bgImg.copy()
		cv2.line(bgImg, armLine['startPt'], armLine['endPt'], (self.lineColor, self.lineColor, self.lineColor), 1)
		# logging.debug('capImg:%s, img:%s' % (capImg.shape, img.shape))
		dst = cv2.add(capImg, bgImg)
		return dst

	def getCoveredPoints(self, img):
		coveredPoints = []
		rows, cols, channels = img.shape
		RGB2Str = lambda clr : '%03d%03d%03d' % (clr[0], clr[1], clr[2])
		for r in range(rows):
			for c in range(cols):
				if RGB2Str(img[r, c]) == '000000000' or RGB2Str(img[r, c]) == '031031031':
					continue

				coveredPoints.append([c, r])
		return coveredPoints

	def getTanPointOnSingleSide(self, angle, left=True):
		# 给定角度，求单侧切点
		prevImg = None
		prevCoveredPoints = None
		for step in range(1, 50):
			if not left:
				step *= -1
			armLine = self.getArmLine(angle, step)
			img = self.addArmLineToCap(armLine)
			coveredPoints = self.getCoveredPoints(img)
			if len(coveredPoints) > 0:
				prevImg = img
				prevCoveredPoints = coveredPoints
			else:
				idx = len(prevCoveredPoints) / 2
				tanPt = prevCoveredPoints[idx]
				# logging.debug(tanPt)
				# cv2.polylines(prevImg, numpy.array([tanPt], numpy.int32).reshape(-1, 1, 2), True, (0, 0, 255), 1)
				# WaitToClose(prevImg)
				return tanPt
		return None

	def GetTanPoints(self, angle):	
		# 给定角度，求左右两个切点，角度是正方向→顺时针，返回的是相对cap中心的偏移
		lpt = self.getTanPointOnSingleSide(angle, left=True)
		rpt = self.getTanPointOnSingleSide(angle, left=False)
		lpt[0] -= self.capX
		lpt[1] -= self.capY
		rpt[0] -= self.capX
		rpt[1] -= self.capY
		return lpt, rpt

class CapUT(unittest.TestCase):
	def setUp(self):
	    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
	    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)

	def test01(self):
		capTPHelper = CapTanPointHelper()
		lpt, rpt = capTPHelper.GetTanPoints(202)

	def test02(self):
		capTPHelper = CapTanPointHelper()
		for angle in range(0, 360):
			lpt, rpt = capTPHelper.GetTanPoints(angle)
			logging.debug('angle:%3d, l:(%3d, %3d), r:(%3d, %3d)' % (angle, lpt[0], lpt[1], rpt[0], rpt[1]))

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main
    # cmd: python -m unittest mpcap.CapUT.test01
