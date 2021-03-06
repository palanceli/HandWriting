
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
import inspect
import time
import timeit

def distance(x1, y1, x2, y2):
	return math.sqrt((y2 - y1)**2 + (x2 - x1)**2)

def WaitToClose(img):
	while True:
		cv2.imshow('image', img)
		if cv2.waitKey(20) & 0xFF == 27:
			break
	
	cv2.destroyAllWindows()

class MPBaseLine(object):
	# MPLine记录一条笔画，其中x y t是原始数据，由该类负责记录和保存，
	def __init__(self):
		self.data = []	# 每个元素是一个[x, y, t] = MPPoint

	def BaseData(self):
		return self.data

	def BaseData2NumpyPts(self):
		pts = numpy.zeros((len(self.data), 2), numpy.int32)
		for i in range(len(self.data)):
			pts[i, 0] = self.data[i][0]
			pts[i, 1] = self.data[i][1]
		return pts

	def Add(self, x, y, t=None):
		if t == None:
			t = timeit.default_timer()
		point = [x, y, t]
		self.data.append(point)
		return point

	def Dumps(self):
		return json.dumps(self.data)

	def Loads(self, string):
		self.data = json.loads(string)

	def ToString(self):
		lineString = ''
		for point in self.data:
			x, y, t = point[0], point[1], point[2]
			lineString += '[%d,%d,%d]' % (x, y, t)
		return lineString

	def FromString(self, lineString):
		pointList = lineString.strip().split('][')
		for point in pointList:
			x, y, t = point.strip('[').strip(']').split(',')
			self.Add(int(x), int(y), int(t))

	def GetPointNum(self):
		return len(self.data)

class OutlineHelper(object):
	@staticmethod
	def IsIntersect(ptA, ptB, ptC, ptD):
		# 线段AB和CD相交的条件为：CD在直线AB的两侧 && AB在直线CD的两侧
		if OutlineHelper.ptIs2SidesLine(ptA, ptB, ptC, ptD) and OutlineHelper.ptIs2SidesLine(ptC, ptD, ptA, ptB):
			return True
		return False

	@staticmethod
	def ptIs2SidesLine(pt0, pt1, line0, line1):
		# line的直线方程为：f(x, y) = (y - yA) * (xA - XB) - (x - xA) * (yA - YB) = 0
		# CD 在直线两侧的条件为：fC * fD > 0
		xC, yC = pt0[0], pt0[1]
		xD, yD = pt1[0], pt1[1]
		xA, yA = line0[0], line0[1]
		xB, yB = line1[0], line1[1]
		fC = (yC - yA) * (xA - xB) - (xC - xA) * (yA - yB)
		fD = (yD - yA) * (xA - xB) - (xD - xA) * (yA - yB)
		if (fC > 0 and fD < 0) or (fC < 0 and fD > 0):
			return True
		return False

	@staticmethod
	def AppendOutline2Pts(olx, oly, bx, by, pts):
		if len(pts) == 0:			# 之前为空，则直接添加
			pts.append((olx, oly, bx, by))
			return
		# len(pts) >= 1
		x0, y0 = olx, oly
		bx0, by0 = bx, by
		x1, y1 = pts[-1][0], pts[-1][1]
		bx1, by1 = pts[-1][2], pts[-1][3]
		if x0 == x1 and y0 == y1:	# 和前一个点重合，则不再重复添加
			return

		# 前一个脊椎-肋骨端子 和当前脊椎-肋骨端子之间如果相交，则不添加
		if OutlineHelper.IsIntersect((bx0, by0), (x0, y0), (bx1, by1), (x1, y1)):
			return

		if len(pts) == 1:
			pts.append((olx, oly, bx, by))
			return
		# len(pts) > 1
		x2 = pts[-2][0]
		y2 = pts[-2][1]
		# 如果∠210是个锐角或直角，则不再添加 pt0
		if ((x2 - x0)**2 + (y2 - y0)**2) <= ((x2 - x1)**2 + (y2 - y1)**2) + ((x1 - x0)**2 + (y1 - y0)**2):
			msg = '发现锐角：(%d, %d), (%d, %d), (%d, %d), %d - %d - %d' % (x0, y0, x1, y1, x2, y2, 
				(x2 - x0)**2 + (y2 - y0)**2, (x2 - x1)**2 + (y2 - y1)**2, (x1 - x0)**2 + (y1 - y0)**2)
			logging.debug(msg)
			return
		pts.append((olx, oly, bx, by))

class MPLine(MPBaseLine):
	def __init__(self):
		MPBaseLine.__init__(self)
		self.skelenton = []	# [{'lpt':, 'rpt':, 'width':}] 每个脊椎对应两根肋骨端子
		self.startCap = None
		self.outline = None
		self.isEnd = False
		self.maxlWidth = 0
		self.maxrWidth = 0

	def fillSkelentonAsBase(self):
		baseLen = len(self.data)
		skelentonLen = len(self.skelenton)
		for i in range(baseLen - skelentonLen):
			self.skelenton.append(None)

	def getSkelentonPts(self, basePt):
		''' 获取与basePt对应的肋骨端子 '''
		index = self.data.index(basePt)
		if index >= len(self.skelenton):
			self.fillSkelentonAsBase()

		return self.skelenton[index]

	def setSkelentonPts(self, basePt, lpt, rpt, lw, rw):
		''' 设置与basePt对应的肋骨端子 '''
		index = self.data.index(basePt)
		if index >= len(self.skelenton):
			self.fillSkelentonAsBase()

		lx, ly = int(lpt[0]), int(lpt[1])
		rx, ry = int(rpt[0]), int(rpt[1])
		if self.skelenton[index] is None:
			self.skelenton[index] = {'lpt':[lx, ly], 'rpt':[rx, ry], 'lw':lw, 'rw':rw}
		else:
			skl = self.skelenton[index]
			skl['lpt'][0] = lx
			skl['lpt'][1] = ly
			skl['rpt'][0] = rx
			skl['rpt'][1] = ry
			skl['lw'] = lw
			skl['rw'] = rw
		return self.skelenton[index]

	def GetOutline(self):
		return self.outline

	def GetSkelenton(self):
		return self.skelenton

	def AddBegin(self, x, y, t=None):
		mpPoint = None
		if len(self.data) == 0:
			self.startCap = Cap(x, y)
			mpPoint = MPBaseLine.Add(self, x, y, t)
		if mpPoint is not None:
			self.skelenton.append(None)
		return mpPoint

	def AddContinue(self, x, y, t=None):
		mpPoint = None
		if len(self.data) == 0:
			mpPoint = self.AddBegin(x, y, t)
		else:
			x1, y1, t1 = self.data[-1][0], self.data[-1][1], self.data[-1][2]
			if x1 == x and y1 == y: # 如果和上一个点重合，则抛弃
				return None
			if t == None:
				t = timeit.default_timer()
			if t - t1 < 0.005:	# 如果两次采样的时间间隔小于5ms，则抛弃，因为计算机在这个范围的精度不准确
				return None

			mpPoint = MPBaseLine.Add(self, x, y, t)
		if mpPoint is not None:
			self.skelenton.append(None)
		self.updateSkelenton()	# 更新排骨架
		self.updateOutline()	# 更新轮廓
		return mpPoint

	def AddEnd(self, x, y, t=None):
		mpPoint = None
		if len(self.data) == 0:
			mpPoint = self.AddBegin(x, y, t)
		else:
			mpPoint = MPBaseLine.Add(self, x, y, t)
		if mpPoint is not None:
			self.skelenton.append(None)
		self.isEnd = True
		return mpPoint

	def Loads(self, string):
		data = json.loads(string)
		for point in data:
			x, y, t = point[0], point[1], point[2]
			if data.index(point) == 0:
				self.AddBegin(x, y, t)
			elif data.index(point) < len(data) - 1:
				self.AddContinue(x, y, t)
			else:
				self.AddEnd(x, y, t)

	def createSkelenton4Base(self, basePt):
		if len(self.data) < 1:	# 只有一个关节，无法计算向量
			return None
		lx, ly, lw, rx, ry, rw = None, None, None, None, None, None
		x, y, t = basePt[0], basePt[1], basePt[2]
		idx = self.data.index(basePt)

		if idx == 0:	# 首节点，肋骨是由运笔角度和cap切点决定
			# logging.debug('首节点')
			pta1 = self.data[idx + 1]	# 取后一个节点
			xa1, ya1, ta1 = pta1[0], pta1[1], pta1[2]
			deltaY = ya1 - y
			deltaX = xa1 - x
			tanInfo = self.startCap.GetTanPointsInfo(deltaX, deltaY)
			# 得到相对于cap中心的位置，
			lx, ly, rx, ry = tanInfo['lx'], tanInfo['ly'], tanInfo['rx'], tanInfo['ry']
			lx += x 	# 转成绝对坐标
			ly += y
			rx += x
			ry += y
			lw = distance(lx, ly, x, y)
			rw = distance(rx, ry, x, y)
			self.maxlWidth = distance(lx, ly, x, y)
			self.maxrWidth = distance(rx, ry, x, y)
			logging.debug('脊柱(%3d,%3d), l肋骨(%3d,%3d), r肋骨(%3d,%3d), lw:%2d, rw:%2d' % 
				(x, y, lx, ly, rx, ry, lw, rw))
			return {'lx':lx, 'ly':ly, 'lw': lw, 'rx':rx, 'ry':ry, 'rw':rw}
		elif idx == len(self.data) - 1:	# 末节点，和普通节点不同在于，它和前一个点（而不是后一个点）确定方向
			# logging.debug('末节点')
			ptb1 = self.data[idx - 1]	# 取前一个节点
			xb1, yb1, tb1 = ptb1[0], ptb1[1], ptb1[2]
			d = distance(x, y, xb1, yb1)
			v = d / ((t - tb1) * 1000)	# 每毫秒移动的像素数
			if idx == 1:	# 补丁：前三个点给速度助力，倾向于收缩
				v += 0.3
			elif idx == 2:
				v += 0.2
			elif idx == 3:
				v += 0.1

			sklb1 = self.getSkelentonPts(ptb1)
			if sklb1 is None:
				logging.debug('前一个skl为None!')
				raise Exception('prev skl is None!')
			lwb1, rwb1 = sklb1['lw'], sklb1['rw']
			# 速度在如下区间的变化率deltaW = (w - wb1) / wb1分别为：
			# (0, 0.6) 	线性由10%~0
			#  0.6		0
			# (0.6, 3)	0~ -10%
			# [3, +∞)	-10%
			deltaW = 0
			if v < 0.6:
				deltaW = (0.6 - v) * 0.2
			elif v > 0.6 and v < 3:
				deltaW = (0.6 - v) * 0.2 / (3 - 1)
			elif v >= 3:
				deltaW = -0.2
			# logging.debug(deltaW)
			if idx >= 2:
				ptb2 = self.data[idx - 2]
				sklb2 = self.getSkelentonPts(ptb2)
				if sklb2 is None:
					logging.debug('前一个skl为None!')
					raise Exception('prev skl is None!')
				lwb2, rwb2 = sklb2['lw'], sklb2['rw']
				# 如果翻转，先缓冲一下
				if (lwb1 - lwb2 > 0 or rwb1 - rwb2 > 0) and deltaW < 0:
					deltaW = 0
				elif (lwb1 - lwb2 < 0 or rwb1 - rwb2 < 0) and deltaW > 0:
					deltaW = 0

			# 应用变化率后，线宽不大于maxWidth，不小于(maxWidth / 4)
			if lwb1 * deltaW > d / 2:
				lw = lwb1 + d / 2
			elif lwb1 * deltaW < - (d / 2):
				lw = lwb1 - d / 2
			else:
				lw = lwb1 * (1 + deltaW)
			# logging.debug(lw)
			lw = min(lw, self.maxlWidth)
			lw = max(lw, self.maxlWidth / 2)
			# logging.debug(lw)

			if rwb1 * deltaW > d / 2:
				rw = rwb1 + d / 2
			elif rwb1 * deltaW < - (d / 2):
				rw = rwb1 - d / 2
			else:
				rw = rwb1 * (1 + deltaW)
			rw = min(rw, self.maxrWidth)
			rw = max(rw, self.maxrWidth / 2)

			lx = x + (y - yb1) * lw / d
			ly = y - (x - xb1) * lw / d
			rx = x - (y - yb1) * rw / d
			ry = y + (x - xb1) * rw / d
			# logging.debug('脊柱(%3d,%3d), l肋骨(%3d,%3d), r肋骨(%3d,%3d), lw:%2d, rw:%2d, v:%5.2f' % 
			# 	(x, y, lx, ly, rx, ry, lw, rw, v))

			return {'lx':lx, 'ly':ly, 'lw': lw, 'rx':rx, 'ry':ry, 'rw':rw}

		# 非首/末节点，当它是末节点的时候已经计算过宽度了，此处只需要更新左右肋骨
		# logging.debug('非首末节点')
		skl = self.getSkelentonPts(basePt)
		if skl is None:
			logging.debug('skl is None!')
			raise Exception('skl is None!')
		lw, rw = skl['lw'], skl['rw']
		pta1 = self.data[idx + 1]
		xa1, ya1 = pta1[0], pta1[1]
		d = distance(x, y, xa1, ya1)
		# logging.debug('(x, y)=(%3d, %3d), (xa1, ya1)=(%3d, %3d), d=%d' % (x, y, xa1, ya1, d))
		lx = x + (ya1 - y) * lw / d
		ly = y - (xa1 - x) * lw / d
		rx = x - (ya1 - y) * lw / d
		ry = y + (xa1 - x) * lw / d

		# logging.debug('lpt:(%d, %d) - %d, rpt:(%d, %d) - %d' % (int(lx), int(ly), lw, rx, ry, rw))

		return {'lx':int(lx), 'ly':int(ly), 'lw':lw, 'rx':int(rx), 'ry':int(ry), 'rw':rw}

	def updateSkelenton(self):
		if len(self.data) < 2: # 如果只有一个关节，无法计算法向量
			return None

		skl = self.createSkelenton4Base(self.data[-2])
		self.setSkelentonPts(self.data[-2], (skl['lx'], skl['ly']), (skl['rx'], skl['ry']), skl['lw'], skl['rw'])
		# logging.debug(skl)
		skl = self.createSkelenton4Base(self.data[-1])
		self.setSkelentonPts(self.data[-1], (skl['lx'], skl['ly']), (skl['rx'], skl['ry']), skl['lw'], skl['rw'])
		# logging.debug(skl)
		return

	def ptsIs2SidesOfLine(self, pt0, pt1, line0, line1):
		# line的直线方程为：f(x, y) = (y - yA) * (xA - XB) - (x - xA) * (yA - YB) = 0
		# CD 在直线两侧的条件为：fC * fD > 0
		xC, yC = pt0[0], pt0[1]
		xD, yD = pt1[0], pt1[1]
		xA, yA = line0[0], line0[1]
		xB, yB = line1[0], line1[1]
		fC = (yC - yA) * (xA - xB) - (xC - xA) * (yA - yB)
		fD = (yD - yA) * (xA - xB) - (xD - xA) * (yA - yB)
		if (fC >= 0 and fD <= 0) or (fC <= 0 and fD >= 0):
			return True
		return False

	def isLineIntersect(self, ptA, ptB, ptC, ptD):
		# 线段AB和CD相交的条件为：CD在直线AB的两侧 && AB在直线CD的两侧
		if self.ptsIs2SidesOfLine(ptA, ptB, ptC, ptD) and self.ptsIs2SidesOfLine(ptC, ptD, ptA, ptB):
			return True
		return False

	def createInterpOutline(self, line):
		xList = []
		yList = []
		for pt in line:
			xList.append(pt[0])
			yList.append(pt[1])

		tck, u = scipy.interpolate.splprep([xList, yList], k=3, s=0)
		xInterp = numpy.linspace(0, 1, num=20, endpoint=True)
		out = scipy.interpolate.splev(xInterp, tck)

		outline = []
		for i in range(len(out[0])):
			outline.append([out[0][i], out[1][i]])
		return outline

	def createSingleSideOutline(self, sideName):
		outline = []	# [(outlineX, outlineY, baseX, baseY)]
		for pt in self.data:
			skl = self.getSkelentonPts(pt)
			if skl is None:
				continue

			baseX, baseY = pt[0], pt[1]
			sideX, sideY = skl[sideName][0], skl[sideName][1]
			idx = self.data.index(pt)
			if idx == 0:	# 如果是首节点，直接插入
				outline.append((sideX, sideY, baseX, baseY))
				continue

			# 前一个轮廓数据，前面已经插入了首节点，所以一定存在前一个节点
			outlineX1, outlineY1 = outline[-1][0], outline[-1][1]	
			baseX1, baseY1 = outline[-1][2], outline[-1][3]
			# 如果当前肋骨和前一根肋骨相交，则不添加
			if self.isLineIntersect((baseX1, baseY1), (outlineX1, outlineY1), (baseX, baseY), (sideX, sideY)):
				continue

			# 如果当前肋骨端子落在cap内，则不添加
			if self.startCap.PtInCap(sideX, sideY):
				continue
			outline.append((sideX, sideY, baseX, baseY))
			# logging.debug('添加轮廓：(%d, %d)' % (sideX, sideY))
		# 多于4个点则采用3次样条插值
		# if len(outline) >= 4:
		# 	return self.createInterpOutline(outline)
		return outline

	def updateOutline(self):
		# 只根据排骨架生成轮廓
		lpts = self.createSingleSideOutline('lpt')
		rpts = self.createSingleSideOutline('rpt')

		outline = numpy.zeros((len(lpts) + len(rpts) + 1, 2), numpy.int32)
		nStart = 0
		
		outline[nStart : nStart + len(lpts), :] = [ (pt[0], pt[1]) for pt in lpts]
		nStart += len(lpts)

		outline[nStart, :] = [self.data[-1][0], self.data[-1][1]]
		nStart += 1

		outline[nStart: , :] = [(pt[0], pt[1]) for pt in rpts[::-1]]
		nStart += len(rpts)

		self.outline = outline

	def Draw2Img(self, img):
		if self.data is None or len(self.data) == 0 or self.skelenton is None or self.outline is None:
			return

		lightColor = (220, 220, 220)
		blackColor = (0, 0, 0)
		fillColor = (31, 31, 31)

		self.startCap.Paste2Img(img)
		# logging.debug(outlinePts)
		cv2.polylines(img, [self.outline], True, fillColor, 1, cv2.LINE_AA)	# 绘制轮廓线
		cv2.polylines(img, self.outline.reshape(-1, 1, 2), True, (0, 0, 255), 2)	# 绘制轮廓点
		# cv2.fillPoly(img, [self.outline], fillColor)
		# cv2.fillConvexPoly(img, self.outline, fillColor)
		cv2.polylines(img, self.BaseData2NumpyPts().reshape(-1, 1, 2), True, (0, 0, 255), 2) # 绘制笔迹节点
		# logging.debug(mpLine.GetBasePoints())

class Cap(object):
	FACTOR = 1
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.img = cv2.resize(cv2.imread('dot.png'), None, fx=Cap.FACTOR, fy=Cap.FACTOR, interpolation=cv2.INTER_CUBIC)
		logging.debug('cap 中心点位置：(%d, %d)' % (self.x, self.y))

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
		x, y = int(x), int(y)
		h, w = int(h), int(w)
		roi = img[y : y+h, x : x+w]
		bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
		dst = cv2.add(bg, fg)
		img[y : y+h, x : x+w] = dst
		return img

	def GetTanPointsInfo(self, deltaX, deltaY):
		capMetaData = CapMetaData()
		capMetaData.Load('capMetaData.json')
		tanInfo = capMetaData.Get(deltaX, deltaY)
		return tanInfo

	def PtInCap(self, x, y):
		(ltx, lty) = self.getLeftTop()
		x = x - ltx
		y = y - lty
		if x < 0 or y < 0:
			return False
		rows, cols, channels = self.img.shape
		if y >= rows or x >= cols:
			return False

		x, y = int(x), int(y)
		if self.img[y, x, 0] == 0 and self.img[y, x, 1] == 0 and self.img[y, x, 2] == 0:
			return False
		return True

class CapTanPointHelper(object):
	''' 线下工具，用于对指定w、h的cap，生成各个角度的切点 '''
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
	''' 存取cap 切点数据 '''
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
		# logging.debug(tan)
		for i in self.data:
			# logging.debug(i['tan'])
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
		# logging.debug(obj)
		if (deltaX >= 0 and deltaY >= 0) or (deltaX <= 0 and deltaY > 0): 	# ↘ ↙
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

		tanInfo = capMetaData.Get(1, -58)
		self.assertEqual(tanInfo['rx'], 17)
		self.assertEqual(tanInfo['ry'], 4)
		self.assertEqual(tanInfo['lx'], -18)
		self.assertEqual(tanInfo['ly'], -20)

	def testa04(self):
		# 测试meta data数据的正确性
		capTPHelper = CapTanPointHelper()
		# capTPHelper.CreateCapMetaFile('capMetaData.json')

		capMetaData = CapMetaData()
		capMetaData.Load('capMetaData.json')
		tanInfo = capMetaData.Get(-1, 2)
		self.assertEqual(tanInfo['lx'], 15)
		self.assertEqual(tanInfo['ly'], 14)
		self.assertEqual(tanInfo['rx'], -18)
		self.assertEqual(tanInfo['ry'], -20)

		tanInfo = capMetaData.Get(1, -2)
		self.assertEqual(tanInfo['rx'], 15)
		self.assertEqual(tanInfo['ry'], 14)
		self.assertEqual(tanInfo['lx'], -18)
		self.assertEqual(tanInfo['ly'], -20)

	def testa05(self):
		# 测试meta data数据的正确性
		capTPHelper = CapTanPointHelper()
		# capTPHelper.CreateCapMetaFile('capMetaData.json')

		capMetaData = CapMetaData()
		capMetaData.Load('capMetaData.json')
		tanInfo = capMetaData.Get(-1, 0)
		self.assertEqual(tanInfo['lx'], 6)
		self.assertEqual(tanInfo['ly'], 20)
		self.assertEqual(tanInfo['rx'], -15)
		self.assertEqual(tanInfo['ry'], -20)

		tanInfo = capMetaData.Get(1, 0)
		self.assertEqual(tanInfo['lx'], -15)
		self.assertEqual(tanInfo['ly'], -20)
		self.assertEqual(tanInfo['rx'], 6)
		self.assertEqual(tanInfo['ry'], 20)

	def testa06(self):
		# 测试meta data数据的正确性
		capTPHelper = CapTanPointHelper()
		# capTPHelper.CreateCapMetaFile('capMetaData.json')

		capMetaData = CapMetaData()
		capMetaData.Load('capMetaData.json')
		tanInfo = capMetaData.Get(1, 1)
		self.assertEqual(tanInfo['lx'], 11)
		self.assertEqual(tanInfo['ly'], -9)
		self.assertEqual(tanInfo['rx'], -1)
		self.assertEqual(tanInfo['ry'], 18)

		tanInfo = capMetaData.Get(-1, -1)
		self.assertEqual(tanInfo['rx'], 11)
		self.assertEqual(tanInfo['ry'], -9)
		self.assertEqual(tanInfo['lx'], -1)
		self.assertEqual(tanInfo['ly'], 18)

	def testm01(self):
		# 需要人眼判断切点位置，
		# 需要把getTanPointOnSingleSide()函数中绘制直线的注释打开
		ctph = CapTanPointHelper()
		deltaX = -1
		deltaY = 58
		tan = float(deltaY) / float(deltaX)
		radians = math.atan(tan)
		degrees = math.degrees(radians)
		lpt, rpt = ctph.calculateTanPoints(degrees)
		logging.debug('(%d, %d), (%d, %d)' % (lpt[0], lpt[1], rpt[0], rpt[1]))

	def testm02(self):
		img = numpy.zeros((300, 300, 3), numpy.uint8)
		img[:, :] = (255, 255, 255)
		cap = Cap(100, 100)
		cap.Paste2Img(img)
		pts = numpy.array([[90, 90], [90, 100], [110, 110], [120, 120], [105, 105]], numpy.int32)
		cv2.polylines(img, pts.reshape(-1, 1, 2), True, (0, 0, 255), 2)
		WaitToClose(img)
		self.assertTrue(cap.PtInCap(90, 90))
		self.assertFalse(cap.PtInCap(90, 100))
		self.assertTrue(cap.PtInCap(110, 110))
		self.assertTrue(cap.PtInCap(105, 105))
		self.assertFalse(cap.PtInCap(120, 120))

class MPLineUT(unittest.TestCase):
	def setUp(self):
	    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
	    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
	    self.getDataDir = lambda : 'utdata/%s/%s' % (self.__class__.__name__, inspect.stack()[1][3])

	def test01(self):
		''' 添加并保存笔迹 '''
		mpLine = MPLine()
		mpLine.AddBegin(85, 174, 337092747352829)
		mpLine.AddContinue(93, 174, 337092749491282)
		mpLine.AddContinue(112, 171, 337092785333021)
		mpLine.AddContinue(130, 169, 337092787032017)
		mpLine.AddContinue(141, 168, 337092791450931)
		mpLine.AddContinue(178, 162, 337092822030929)
		mpLine.AddContinue(188, 160, 337092851645709)
		mpLine.AddContinue(224, 155, 337092853421859)
		mpLine.AddContinue(231, 154, 337092856345850)
		mpLine.AddContinue(253, 152, 337092883426102)
		mpLine.AddContinue(257, 151, 337092889972152)
		mpLine.AddContinue(268, 151, 337092918292494)
		mpLine.AddContinue(270, 151, 337092921288455)
		mpLine.AddEnd(272, 151, 337092948137001)
		jstring = mpLine.Dumps()

		if not os.path.exists(self.getDataDir()):
			os.makedirs(self.getDataDir())
		jsonPath = '%s/mpLine.json' % (self.getDataDir())
		with open(jsonPath, 'wb') as f:
			f.write(jstring)

	def test02(self):
		''' 读取并绘制笔迹 '''
		jstring = ''
		jsonPath = '%s/mpLine.json' % (self.getDataDir())
		with open(jsonPath, 'rb') as f:
			jstring = f.read()

		mpLine = MPLine()
		mpLine.Loads(jstring)

		img = numpy.zeros((500, 800, 3), numpy.uint8)
		img[:, :] = (255, 255, 255)
		# 绘制脊柱
		basePts = mpLine.BaseData2NumpyPts()
		# logging.debug(basePts)
		cv2.polylines(img, [basePts.reshape(-1, 1, 2)], False, (0, 0, 0), 1)
		WaitToClose(img)

	def test03(self):
		''' 读取笔迹，生成并绘制骨架 '''
		jstring = ''
		jsonPath = '%s/mpLine.json' % (self.getDataDir())
		with open(jsonPath, 'rb') as f:
			jstring = f.read()

		mpLine = MPLine()
		mpLine.Loads(jstring)

		img = numpy.zeros((500, 800, 3), numpy.uint8)
		img[:, :] = (255, 255, 255)
		# 绘制脊柱
		basePts = mpLine.BaseData2NumpyPts()
		# logging.debug(basePts)
		cv2.polylines(img, [basePts.reshape(-1, 1, 2)], False, (0, 0, 0), 1)
		# 绘制肋骨
		basePts = mpLine.BaseData2NumpyPts()
		skelentons = mpLine.GetSkelenton()
		for skl in skelentons:
			if skl is None:
				continue
			# logging.debug(skl)
			lx, ly = skl['lpt'][0], skl['lpt'][1]
			rx, ry = skl['rpt'][0], skl['rpt'][1]
			cv2.line(img, (lx, ly), (rx, ry), (0, 0, 0), 1)
		WaitToClose(img)

	def test04(self):
		''' 跟踪鼠标轨迹，只记录轨迹点，不做多余运算，和test04作对比（只能绘制一笔） '''
		jsonPath = '%s/mpLine.json' % (self.getDataDir())
		img = numpy.zeros((500, 800, 3), numpy.uint8)
		img[:, :] = (255, 255, 255)
		cv2.imshow('img', img)
		
		mpLine = []

		# 将起笔、运笔、抬笔交给画笔处理
		def mouseCallback(event, x, y, flags, param):
			app = param
			if event == cv2.EVENT_LBUTTONDOWN:
				mpLine.append((x, y, timeit.default_timer()))
			elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
				x1, y1, t1 = mpLine[-1][0], mpLine[-1][1], mpLine[-1][2]
				if x1 == x and y1 == y:
					return
				t = timeit.default_timer()
				if t - t1 < 0.005:
					return
				mpLine.append((x, y, t))

			elif event == cv2.EVENT_LBUTTONUP:
				x1, y1 = mpLine[-1][0], mpLine[-1][1]
				if not (x1 == x and y1 == y):
					mpLine.append((x, y, timeit.default_timer()))

				pts = []
				for pt in mpLine:
					x, y, t = pt[0], pt[1], pt[2]
					pts.append([x, y])
					if mpLine.index(pt) == 0:
						logging.debug('(x, y)=(%3d, %3d), t=%13.6f' % (x, y, t))
					else:
						idx = mpLine.index(pt)
						x1, y1, t1 = mpLine[idx - 1][0], mpLine[idx - 1][1], mpLine[idx - 1][2]
						d = distance(x1, y1, x, y)
						deltaT = (t - t1) * 1000
						v = d / deltaT
						logging.debug('(x, y)=(%3d, %3d), t=%13.6f, v=%5.2f, d=%d, deltaT=%5.2f(ms)' % 
							(x, y, t, v, d, deltaT))
				img[:, :] = (255, 255, 255)

				pts = numpy.array(pts, numpy.int32)
				cv2.polylines(img, pts.reshape(-1, 1, 2), True, (0, 0, 0), 3)
				jstring = json.dumps(mpLine)
				with open(jsonPath, 'w') as f:
					f.write(jstring)

		cv2.setMouseCallback('img', mouseCallback, self)

		# 响应快捷键
		while True:
			cv2.imshow('img', img)
			pressedKey = cv2.waitKey(10)	# 等待10ms，如果无按键，返回-1
			if pressedKey == -1:
				continue

			if pressedKey & 0xFF == 27:	# ESC 	退出
				break

		cv2.destroyAllWindows()

	def test05(self):
		''' 跟踪鼠标轨迹，生成MPLine，并保存。（只能绘制一笔） '''
		jsonPath = '%s/mpLine.json' % (self.getDataDir())
		img = numpy.zeros((500, 800, 3), numpy.uint8)
		img[:, :] = (255, 255, 255)
		cv2.imshow('img', img)
		
		mpLine = MPLine()

		# 将起笔、运笔、抬笔交给画笔处理
		def mouseCallback(event, x, y, flags, param):
			app = param
			if event == cv2.EVENT_LBUTTONDOWN:
				mpLine.AddBegin(x, y)
				mpLine.Draw2Img(img)
			elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
				mpLine.AddContinue(x, y)
				img[:, :] = (255, 255, 255)
				mpLine.Draw2Img(img)
			elif event == cv2.EVENT_LBUTTONUP:
				mpLine.AddEnd(x, y)
				img[:, :] = (255, 255, 255)
				mpLine.Draw2Img(img)
				jstring = mpLine.Dumps()
				with open(jsonPath, 'w') as f:
					f.write(jstring)

		cv2.setMouseCallback('img', mouseCallback, self)

		# 响应快捷键
		while True:
			cv2.imshow('img', img)
			pressedKey = cv2.waitKey(10)	# 等待10ms，如果无按键，返回-1
			if pressedKey == -1:
				continue

			if pressedKey & 0xFF == 27:	# ESC 	退出
				break

		cv2.destroyAllWindows()

	def test06(self):
		''' 读取笔迹数据，生成并绘制轮廓 '''
		jstring = ''
		jsonPath = '%s/mpLine.json' % (self.getDataDir())
		with open(jsonPath, 'rb') as f:
			jstring = f.read()

		mpLine = MPLine()
		mpLine.Loads(jstring)

		# logging.debug('mpLine skelenton:%s' % (mpLine.skelenton))
		img = numpy.zeros((500, 800, 3), numpy.uint8)
		img[:, :] = (255, 255, 255)
		mpLine.Draw2Img(img)
		WaitToClose(img)


if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main
    # cmd: python -m unittest mpcap.CapUT.test01
