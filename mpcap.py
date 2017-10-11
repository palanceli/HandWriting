
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
import json

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
				cv2.polylines(prevImg, numpy.array([tanPt], numpy.int32).reshape(-1, 1, 2), True, (0, 0, 255), 1)
				WaitToClose(prevImg)
				return tanPt
		return None

	def calculateTanPoints(self, angle):	
		# 给定角度，求左右两个切点，角度是正方向→顺时针，返回的是相对cap中心的偏移
		lpt = self.getTanPointOnSingleSide(angle, left=True)
		rpt = self.getTanPointOnSingleSide(angle, left=False)
		lpt[0] -= self.capX
		lpt[1] -= self.capY
		rpt[0] -= self.capX
		rpt[1] -= self.capY
		return lpt, rpt

	def CreateCapMetaFile(self, path):
		# 把数据保存到path，每行格式为：
		# 运笔角度的正切 运笔角度度数 该角度下的左切点 该角度下的右切点
		capDataFile = CapMetaData()
		for angle in range(0, 180):
			lpt, rpt = self.calculateTanPoints(angle)
			capDataFile.Set(angle, lpt, rpt)
		capDataFile.Save(path)

class CapMetaData(object):
	def __init__(self):
		self.data = []
		self.infinitTan = 9999	# tan(90°)

	def Set(self, angle, lpt, rpt):
		tan = None
		if angle == 90:
			tan = self.infinitTan
		else:
			tan = math.tan(math.radians(angle))
		self.data.append({'tan': tan, 'lpt':lpt, 'rpt':rpt})

	def Get(self, deltaX, deltaY):
		tan = float(deltaY) / float(deltaX)
		prev = None
		obj = None
		logging.debug(tan)
		for i in self.data:
			logging.debug(i['tan'])
			if tan == i['tan']:
				obj = i
				break
			elif tan < i['tan']:
				if prev is not None:
					obj = prev
					break
				else:
					obj = i
					break
			else:
				prev = i
		lx = obj['lpt'][0]
		ly = obj['lpt'][1]
		rx = obj['rpt'][0]
		ry = obj['rpt'][1]
		logging.debug(obj)
		if (deltaX >= 0 and deltaY >= 0) or (deltaX <= 0 and deltaY >= 0): 	# ↘ ↙
			return {'lx':lx, 'ly':ly, 'rx':rx, 'ry':ry}
		else:																# ↖ ↗
			return {'lx':rx, 'ly':ry, 'rx':lx, 'ry':ly}

	def Save(self, path):
		def cmp(x, y):
			if x['tan'] < y['tan']:
				return -1
			elif x['tan'] > y['tan']:
				return 1
			return 0

		self.data.sort(cmp)	
		with open(path, 'wb') as f:
			f.write(json.dumps(self.data))

	def Load(self, path):
		with open(path, 'rb') as f:
			self.data = json.loads(f.read())

class CapUT(unittest.TestCase):
	def setUp(self):
	    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
	    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)

	def testa01(self):
		# 测试 角度->切点 的计算结果
		capTPHelper = CapTanPointHelper()
		lpt, rpt = capTPHelper.calculateTanPoints(123)	
		self.assertEqual(lpt[0], 14)
		self.assertEqual(lpt[1], 16)
		self.assertEqual(rpt[0], -18)
		self.assertEqual(rpt[1], -20)

	def testa02(self):
		# 生成meta data数据
		capTPHelper = CapTanPointHelper()
		capTPHelper.CreateCapMetaFile('capMetaData.json')

	def testa03(self):
		# 测试meta data数据的正确性
		capTPHelper = CapTanPointHelper()
		# capTPHelper.CreateCapMetaFile('capMetaData.json')

		capMetaData = CapMetaData()
		capMetaData.Load('capMetaData.json')
		tanInfo = capMetaData.Get(-1, 58)
		self.assertEqual(tanInfo['lx'], 17)
		self.assertEqual(tanInfo['ly'], 4)
		self.assertEqual(tanInfo['rx'], -18)
		self.assertEqual(tanInfo['ry'], -20)

	def testm01(self):
		# 需要人眼判断切点位置，
		# 需要把getTanPointOnSingleSide()函数中绘制直线的注释打开
		ctph = CapTanPointHelper()
		deltaX = -1
		deltaY = 58
		tan = float(deltaY) / float(deltaX)
		radians = math.atan(tan)
		degrees = math.degrees(radians)
		ctph.calculateTanPoints(degrees)

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main
    # cmd: python -m unittest mpcap.CapUT.test01
