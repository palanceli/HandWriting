
# -*- coding:utf-8 -*-

import logging
import os
import numpy
import cv2
import math
import sys
import scipy
import scipy.interpolate

def distance(x1, y1, x2, y2):
	return math.sqrt((y2 - y1)**2 + (x2 - x1)**2)

class MPLine(object):
	# MPLine记录一条笔画，其中x y t是原始数据，由该类负责记录和保存，
	# extra是附加数据，也是和笔迹相关的数据，由调用方负责产生，MPLine只负责记录
	def __init__(self):
		# 每个元素是一个[x, y, t, extra] = MPPoint
		# x y t是基本值，extra是由笔迹效果根据基本值计算出来的
		self.data = []
		self.extra = None
		self.isEnd = False

	def Add(self, x, y, t=None):
		if t == None:
			t = cv2.getTickCount()
		point = [x, y, t, None]
		self.data.append(point)
		return point

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

	def GetBasePoints(self):
		basePoints = numpy.array([(pt[0], pt[1]) for pt in self.data], numpy.int32)
		return basePoints

class MPTracker(object):
	# 笔迹是由笔画（线）组成，笔画由点组成，MPTracker保存笔画的集合
	def __init__(self):
		# 这是一个笔迹数组，每个元素都是MPLine
		self.trackList = []
		self.filePath = 'originTracker.data'

	def AddBegin(self, x, y, t):
		newLine = MPLine()
		newLine.Add(x, y, t)
		self.trackList.append(newLine)
		return newLine

	def AddContinue(self, x, y, t):
		if len(self.trackList) == 0:
			return self.AddBegin(x, y, t)
		lastLine = self.trackList[-1]
		# 如果x,y和前一个点坐标一致，就不再加了
		if lastLine.GetPointNum() > 0:
			lastPoint = lastLine.data[-1]
			if lastPoint[0] == x and lastPoint[1] == y:
				return lastLine
		lastLine.Add(x, y, t)
		return lastLine

	def AddEnd(self, x, y, t):
		lastLine = self.AddContinue(x, y, t)
		lastLine.isEnd = True
		return lastLine

	def Clean(self):
		self.trackList = []

	def Save(self):
		with open(self.filePath, 'wb') as f:
			for line in self.trackList:
				f.write(line.ToString())
				f.write('\n')

	def Load(self):
		self.Clean()
		with open(self.filePath, 'rb') as f:
			for line in f:
				newLine = MPLine()
				newLine.FromString(line)
				if len(newLine.data) > 0:
					self.trackList.append(newLine)

class MagicPenConf(object):
	# 记录配置
	def __init__(self):
		self.conf = {'showOrigin' : True, 'showOriginLine' : True, 'showPolyLine' : True,
		'showCTan':False}

	def set(self, key, value):
		self.conf[key] = value

	def get(self, key):
		return self.conf[key]

class MagicPen(object):
	# MagicPen相当于一个抽象类，只负责记录原始笔迹，以及绘制原始笔迹
	def __init__(self, img, imgName, conf):
		self.mpTracker = MPTracker()
		self.img = img
		self.imgName = imgName
		self.conf = conf

	# {{ Begin、Continue、End 负责记录原始笔迹 }}
	def Begin(self, x, y, t = None):
		return self.mpTracker.AddBegin(x, y, t)

	def Continue(self, x, y, t=None):
		return self.mpTracker.AddContinue(x, y, t)

	def End(self, x, y, t=None):
		return self.mpTracker.AddEnd(x, y, t)

	def Clean(self):
		self.mpTracker.Clean()
		self.img[:, :] = (255, 255, 255)

	def Redraw(self):
		return

	def GetCurrDarwingMPLine(self):
		if len(self.mpTracker.trackList) == 0:
			return None
		mpLine = self.mpTracker.trackList[-1]
		if mpLine.isEnd:
			return None
		return mpLine

	# SaveTrack、LoadTrack 保存和加载原始笔迹
	def SaveTrack(self):
		self.mpTracker.Save()

	def LoadTrack(self):
		mpTracker = MPTracker()
		mpTracker.Load()
		return mpTracker

class MPBrushPointExtra(object):
	# 附着在每个MPPoint 的extra数据
	def __init__(self, lx, ly, rx, ry, width):
		self.data = {'lx':int(lx), 'ly':int(ly), 'rx':int(rx), 'ry':int(ry), 'width':width, 'cap':None}

	def GetLR(self):
		return self.data['lx'], self.data['ly'], self.data['rx'], self.data['ry'], 

	def GetWidth(self):
		return self.data['width']

	def SetCap(self, cap):
		self.data['cap'] = cap

	def GetCap(self):
		return self.data['cap']

class SkelentonHelper(object):
	def MakeSkelenton(self, mpLine):
		# 生成排骨架
		# 根据(x0, y0, t0)、(x1, y1, t1)和(x2, y2, t2, extra2)计算extra1
		return None

	def MakeOutline(self, mpLine):
		# 生成插值线条
		return None

class LineSegmentIntersectJudger(object):
	# 判断两条线段是否相交
	@staticmethod
	def IsIntersect(ptA, ptB, ptC, ptD):
		# 线段AB和CD相交的条件为：CD在直线AB的两侧 && AB在直线CD的两侧
		cls = LineSegmentIntersectJudger
		if cls.ptIs2SidesLine(ptA, ptB, ptC, ptD) and cls.ptIs2SidesLine(ptC, ptD, ptA, ptB):
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

class LinearSkelentonHelper(SkelentonHelper):
	def minWidth(self):
		return 5

	def maxWidth(self):
		return 15

	def calcWidth(self, v):
		# 使用一次线性函数计算线宽
		# (maxWidth - minWidth) / V = (15 - width)/v
		# width = 15 - 5 * x / 2
		V = 4
		if v > V:
			return self.minWidth()
		return 15 - 5 * v / 2

	def MakeSkelenton(self, mpLine):
		if mpLine is None:
			return None
		if len(mpLine.data) < 2: # 如果只有一个关节，无法计算法向量
			return None

		pt0 = mpLine.data[-1]	# 最后一个节点
		x0, y0, t0 = pt0[0], pt0[1], pt0[2]
		pt1 = mpLine.data[-2]	# 倒数第二个节点
		x1, y1, t1 = pt1[0], pt1[1], pt1[2]
		d1 = math.sqrt((y1 - y0)**2 + (x1 - x0)**2)	# pt1 -> pt0的距离
		v1 = d1 * 1000000 / (t0 - t1)				# pt1 -> pt0的速度
		w1 = self.calcWidth(v1)

		if len(mpLine.data) >= 3:
			pt2 = mpLine.data[-3]
			w2 = pt2[3].GetWidth()
			if (w1 - w2) / w2 > 0.1:
				w1 = 1.1 * w2
			elif (w1 - w2) / w2 < -0.1:
				w1 = 0.9 * w2


		lx1 = x1 + (y0 - y1) * w1 / d1
		ly1 = y1 + (x1 - x0) * w1 / d1
		rx1 = x1 + (y1 - y0) * w1 / d1
		ry1 = y1 + (x0 - x1) * w1 / d1
		extraData = MPBrushPointExtra(lx1, ly1, rx1, ry1, w1)
		pt1[3] = extraData 		# 给前一个采样点设置extraData

		if mpLine.data.index(pt1) == 0:		# 为头结点设置CapStart
			capStart = CapStart((x1, y1), (lx1, ly1), (rx1, ry1))
			# logging.debug(capStart.capImg)
			extraData.SetCap(capStart)
		return extraData

	def addPt2Pts(self, pt, pts):
		if len(pts) == 0:			# 之前为空，则直接添加
			pts.append(pt)
			return
		# len(pts) >= 1
		x0 = pt[0]
		y0 = pt[1]
		bx0 = pt[2]
		by0 = pt[3]
		x1 = pts[-1][0]
		y1 = pts[-1][1]
		bx1 = pts[-1][2]
		by1 = pts[-1][3]
		if x0 == x1 and y0 == y1:	# 和前一个点重合，则不再重复添加
			return

		# 前一个脊椎-肋骨端子 和当前脊椎-肋骨端子之间如果相交，则不添加
		if LineSegmentIntersectJudger.IsIntersect((bx0, by0), (x0, y0), (bx1, by1), (x1, y1)):
			return

		if len(pts) == 1:
			pts.append(pt)
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
		pts.append(pt)

	def makeSkelentonOutline(self, mpLine):
		# 不插值，只根据排骨架生成平滑轮廓
		lpts = []
		rpts = []

		for pt in mpLine.data:
			extra = pt[3]
			if extra is None:
				continue
			x, y = pt[0], pt[1]
			lx, ly, rx, ry = extra.GetLR()

			self.addPt2Pts((lx, ly, x, y), lpts)
			self.addPt2Pts((rx, ry, x, y), rpts)

		sCapPts = []
		eCapPts = []
		# 插值首尾Cap
		if len(mpLine.data) > 0:
			ptExtraData = mpLine.data[0][3]
			if ptExtraData is not None:
				capStart = ptExtraData.GetCap()
				if capStart is not None:
					if capStart.GetType() == 1:		# 横
						anchorsStart = capStart.Anchors()
						sCapPts.append(anchorsStart[0])
						sCapPts.append(anchorsStart[1])

			ptExtraData = mpLine.data[-1][3]
			if ptExtraData is not None:
				capEnd = ptExtraData.GetCap()
				if capEnd is not None:
					if capEnd.GetType() == 1:		# 横
						anchorsEnd = capEnd.Anchors()
						eCapPts.append(anchorsEnd[0])
						eCapPts.append(anchorsEnd[1])

		outline = numpy.zeros((len(sCapPts) + len(lpts) + len(rpts) + 1, 2), numpy.int32)
		nStart = 0
		if len(sCapPts) > 0:
			outline[:len(sCapPts), :] = [(pt[0], pt[1]) for pt in sCapPts]
			nStart += len(sCapPts)
		
		outline[nStart : nStart + len(lpts), :] = [ (pt[0], pt[1]) for pt in lpts]
		nStart += len(lpts)

		outline[nStart, :] = [mpLine.data[-1][0], mpLine.data[-1][1]]
		nStart += 1

		outline[nStart: , :] = [(pt[0], pt[1]) for pt in rpts[::-1]]
		nStart += len(rpts)

		mpLine.extra = outline
		return mpLine.extra

	def createCubicInterp(self, x, y):
		tck, u = scipy.interpolate.splprep([x,y], k=3, s=0)
		xInterp = numpy.linspace(0, 1, num=20, endpoint=True)
		out = scipy.interpolate.splev(xInterp, tck)
		xInterps = out[0]
		yInterps = out[1]
		return xInterps, yInterps
		
	def makeInterpOutline(self, mpLine):
		# 通过插值生成更平滑的轮廓
		lxList = []
		lyList = []
		rxList = []
		ryList = []
		if mpLine is None:
			return None

		lastlpt = None
		lastrpt = None
		for pt in mpLine.data:
			ptExtra = pt[3]
			if ptExtra is None:
				continue
			lx, ly, rx, ry = ptExtra.GetLR()
			if (lastlpt is not None) and (lastlpt[0] != lx) and (lastlpt[1] != ly):
				lxList.append(lx)
				lyList.append(ly)
			lastlpt = (lx, ly)
			if (lastrpt is not None) and (lastrpt[0] != rx) and (lastrpt[1] != ry):
				rxList.append(rx)
				ryList.append(ry)
			lastrpt = (rx, ry)
		if len(lxList) < 4 or len(rxList) < 4:
			return None
		# logging.debug(lxList)
		# logging.debug(lyList)
		lxInterps, lyInterps = self.createCubicInterp(lxList, lyList) 	# 左边界插值

		# logging.debug(rxList)
		# logging.debug(ryList)
		rxInterps, ryInterps = self.createCubicInterp(rxList, ryList)	# 右边界插值
		
		# 生成封闭轮廓
		outline = numpy.zeros((len(lxInterps) + len(rxInterps) + 1, 2), numpy.float32)

		outline[ : len(lxInterps), 0] = lxInterps
		outline[ : len(lyInterps), 1] = lyInterps
		outline[len(lxInterps), 0] = mpLine.data[-1][0]
		outline[len(lyInterps), 1] = mpLine.data[-1][1]
		outline[len(lxInterps) + 1 : , 0] = rxInterps[::-1]
		outline[len(lyInterps) + 1 : , 1] = ryInterps[::-1]

		mpLine.extra = outline
		return mpLine.extra

	def MakeOutline(self, mpLine):
		return self.makeSkelentonOutline(mpLine)

class Cap(object):
	def __init__(self):
		self.capImg = None
		self.offsetX = None
		self.offsetY = None
		self.type = None

	def GetType(self):
		return self.type

	def anchors(self):
		rows, cols, channels = self.capImg.shape
		anchors = [
			[cols * 0.65, rows * 0.6], 	# 0 中心点
			[0, 0], 								# 1 左上尖
			[cols * 0.4 - 1, rows * 0.07 + 1], 	# 2 颈
			[cols * 0.8 - 1, rows * 0.25 + 1],	# 3 肩
			[cols * 1.0 - 1, rows * 0.6],		# 4 右
			[cols * 0.95 - 1, rows * 0.8],	# 5 臀
			[cols * 0.65, rows * 1.0 - 1],	# 6 底
			[cols * 0.48, rows * 0.95 - 1],	# 7 左下尖
			[cols * 0.32 + 1, rows * 0.6]]	# 8 腹

		return anchors

	def resize(self, img, width = None, height = None):
		factor = None
		rows, cols, channels = img.shape
		if width is not None:
			factor = float(width) / float(cols)
		if height is not None:
			factor = float(height) / float(rows)
		if factor is None:
			raise Exception('factor is None.')

		return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
		
	def Paste2Img(self, img):
		capgray = cv2.cvtColor(self.capImg, cv2.COLOR_BGR2GRAY)
		ret, mask = cv2.threshold(capgray, 20, 255, cv2.THRESH_BINARY)
		mask_inv = cv2.bitwise_not(mask)
		fg = cv2.bitwise_and(self.capImg, self.capImg, mask=mask)

		rows, cols, channels = self.capImg.shape
		roi = img[self.offsetY : self.offsetY+rows, self.offsetX : self.offsetX+cols]
		bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
		dst = cv2.add(bg, fg)
		img[self.offsetY : self.offsetY+rows, self.offsetX : self.offsetX+cols] = dst
		return img

class CapEnd(Cap):
	def __init__(self, type, basePt, lPt, rPt):
		Cap.__init__(self)
		self.type = type
		img = cv2.imread('dot.png')
		h0, w0, channels = img.shape 	# 原始尺寸
		if self.type == 1:
			h1 = rPt[1] - lPt[1] 		# 预期高度
			self.capImg = self.resize(img, width=None, height=h1)
			self.offsetX = int(lPt[0])
			self.offsetY = int(lPt[1])
		else:
			logging.debug('To Do ...')

	def Anchors(self):
		anchors = self.anchors()
		if self.type == 0:	# 中心点
			return [[anchors[i][0] + self.offsetX, anchors[i][1] + self.offsetY] for i in (0, )]
		elif self.type == 1:	# 横
			return [[anchors[i][0] + self.offsetX, anchors[i][1] + self.offsetY] for i in (0, 6)]
		else:
			logging.debug('To Do ...')

class CapStart(Cap):
	def __init__(self, basePt, lPt, rPt):
		Cap.__init__(self)
		# basePt为脊柱关节，lPt、rPt为左右肋骨端点
		self.type = self.getStartType(basePt, lPt, rPt)
		logging.debug(self.type)
		img = cv2.imread('dot.png')
		h0, w0, channels = img.shape 	# 原始尺寸
		if self.type == 1:
			h1 = rPt[1] - lPt[1] 			# 预期高度
			self.capImg = self.resize(img, width=None, height=h1)
			anchors = self.anchors()
			self.offsetX = int(rPt[0] - anchors[6][0])
			self.offsetY = int(rPt[1] - anchors[6][1])
		else:
			logging.error('To Do ...')

	def getStartType(self, basePt, lPt, rPt):
		# (横：1)，(竖：2)， (撇：3)，(捺：4)，(勾：5)
		deltaY = rPt[1] - lPt[1]
		deltaX = rPt[0] - lPt[0]
		logging.debug('deltaX:%d, deltaY:%d' % (deltaX, deltaY))
		if deltaX == 0: 	# 水平情况
			if deltaY < 0:
				return 3 	# 向左，撇
			return 1 		# 向右，横

		tan = abs(float(deltaY) / float(deltaX))
		if deltaY <= 0 and deltaX > 0:
			if tan < 0.25: 
				return 2 	# (0°~15°)竖↓
			elif tan >= 0.25 and tan < 3.8:
				return 4 	# [15°~75°)捺↘
			return 1 		# [75°~ 90°)横→

		if deltaY <= 0 and deltaX < 0:
			if tan < 3.8:
				return 5 	# (0°~75°)勾↑
			return 1 		# [75°~ 90°)横→

		if deltaY >= 0 and deltaX > 0:
			if tan < 0.25:
				return 2 	# (0°~15°)竖↓
			elif tan >= 0.25 and tan < 3.8:
				return 3 	# [15°~ 90°]撇↙←
			return 5 		# (0°~90°)勾↑←

		if deltaY >= 0 and deltaX < 0:
			return 5 		# (0°~90°)勾↑←

	def Anchors(self):
		anchors = self.anchors()
		if self.type == 0:	# 中心点
			return [[anchors[i][0] + self.offsetX, anchors[i][1] + self.offsetY] for i in (0, )]
		elif self.type == 1:	# 横
			return [[anchors[i][0] + self.offsetX, anchors[i][1] + self.offsetY] for i in (0, 1, 6)]
		elif self.type == 2:	# 竖
			return [[anchors[i][0] + self.offsetX, anchors[i][1] + self.offsetY] for i in (0, 4, 8)]
		elif self.type == 3:	# 撇
			return [[anchors[i][0] + self.offsetX, anchors[i][1] + self.offsetY] for i in (0, 7, 8)]
		elif self.type == 4:	# 捺
			return [[anchors[i][0] + self.offsetX, anchors[i][1] + self.offsetY] for i in (0, 3, 7)]

class MagicPenBrush(MagicPen):
	def __init__(self, img, imgName, conf):
		MagicPen.__init__(self, img, imgName, conf)
		# 已经抬笔的笔画保存到backImg，每次重绘时使用self.img = self.backImg + 尚未抬笔的笔画
		self.backImg = self.img.copy()
		self.maskImg = numpy.zeros((self.backImg.shape[0] + 2, self.backImg.shape[1] + 2), numpy.uint8)
		self.maskImg[:] = 0
		self.skelentonHelper = LinearSkelentonHelper()

	def Clean(self):
		MagicPen.Clean(self)
		self.backImg[:, :] = (255, 255, 255)

	def Begin(self, x, y, t=None):
		# logging.debug('beging:(%d, %d)' % (x, y))
		if len(self.mpTracker.trackList) > 0:	# 如果前一笔没有End，则将其设置为完结，并绘制到backImg上
			mpLine = self.mpTracker.trackList[-1]
			mpLine.isEnd = True
			self.drawMPLineToImg(mpLine, self.backImg)
		return MagicPen.Begin(self, x, y, t)

	def Continue(self, x, y, t=None):
		mpLine = self.GetCurrDarwingMPLine()
		if mpLine is not None and len(mpLine.data) >= 2:
			x1, y1 = mpLine.data[-1][0], mpLine.data[-1][1]
			x2, y2 = mpLine.data[-2][0], mpLine.data[-2][1]
			if x == x1 and y == y1:		# 如果和前一个点重合，则不再重复添加
				return mpLine

			# 如果∠210是个锐角或直角 
			if ((x2 - x)**2 + (y2 - y)**2) <= ((x2 - x1)**2 + (y2 - y1)**2) + ((x1 - x)**2 + (y1 - y)**2):
				msg = '主笔迹发现锐角：(%d, %d), (%d, %d), (%d, %d)，%d - %d - %d\n'% (x, y, x1, y1, x2, y2, 
					(x2 - x)**2 + (y2 - y)**2, (x2 - x1)**2 + (y2 - y1)**2, (x1 - x)**2 + (y1 - y)**2)
				msg += '%s' % mpLine.data
				logging.debug(msg)
				return self.Begin(x, y)

		# logging.debug('continue:(%d, %d)' % (x, y))
		# 记录原始笔迹
		lastLine = MagicPen.Continue(self, x, y, t)

		# 为每个笔迹点生成排骨架
		pointExtraData = self.skelentonHelper.MakeSkelenton(lastLine)

		# 插值生成轮廓
		lineExtraData = self.skelentonHelper.MakeOutline(lastLine)
		return lastLine

	def drawMPLineToImg(self, mpLine, img):
		if mpLine is None or mpLine.data is None:
			return
		if  len(mpLine.data) == 0:
			return
		if  mpLine.extra is None:
			return

		lightColor = (220, 220, 220)
		blackColor = (0, 0, 0)
		fillColor = (31, 31, 31)
		outlinePts = numpy.int32(mpLine.extra)
		# logging.debug(outlinePts)
		cv2.polylines(img, [outlinePts], True, fillColor, 1, cv2.LINE_AA)	# 绘制轮廓
		cv2.fillPoly(img, [outlinePts], fillColor)
		
		# 绘制起笔
		ptExtraData = mpLine.data[0][3]
		capStart = ptExtraData.GetCap()
		# if capStart is not None:
			# capStart.Paste2Img(img)

		# cv2.fillConvexPoly(img, outlinePts, fillColor)
		cv2.polylines(img, outlinePts.reshape(-1, 1, 2), True, blackColor, 3)	# 绘制轮廓节点
		cv2.polylines(img, mpLine.GetBasePoints().reshape(-1, 1, 2), True, (0, 0, 255), 3) # 绘制笔迹节点
		# logging.debug(mpLine.GetBasePoints())

	def End(self, x, y, t=None):
		lastLine = self.Continue(x, y, t)
		# self.createLineExtraData(lastLine)
		# 抬笔的时候把最后一笔画到backImg上
		self.drawMPLineToImg(lastLine, self.backImg)
		# logging.debug('抬笔')

	def Redraw(self):
		numpy.copyto(self.img, self.backImg)
		if self.mpTracker.trackList is None:
			return

		if len(self.mpTracker.trackList) == 0:
			return

		lastLine = self.mpTracker.trackList[-1]
		if not lastLine.isEnd:	# 尚未抬笔的笔画，绘制到self.img上
			self.drawMPLineToImg(lastLine, self.img)

	def LoadTrack(self):
		mpTracker = MagicPen.LoadTrack(self)
		# load extra 
		for mpLine in mpTracker.trackList:
			for mpPoint in mpLine.data:
				x, y, t = mpPoint[0], mpPoint[1], mpPoint[2]
				if mpLine.data.index(mpPoint) == 0:
					self.Begin(x, y, t)
				elif mpLine.data.index(mpPoint) == len(mpLine.data) - 1:
					self.End(x, y, t)
				else:
					self.Continue(x, y, t)

class MagicPenApp(object):
	def __init__(self):
		# 负责创建全局资源
		self.img = self.createImg()	# 画布
		self.imgName = 'image'
		self.conf = MagicPenConf()	# 配置
		self.pen = MagicPenBrush(self.img, self.imgName, self.conf)	# 画笔

	def createImg(self):
		img = numpy.zeros((500, 899, 3), numpy.uint8)
		img[:, :] = (255, 255, 255)
		return img

	def MainProc(self):
		cv2.imshow(self.imgName, self.img)
		cv2.moveWindow(self.imgName, 100, 100)

		# 将起笔、运笔、抬笔交给画笔处理
		def mouseCallback(event, x, y, flags, param):
			app = param
			if event == cv2.EVENT_LBUTTONDOWN:
				app.pen.Begin(x, y)
				app.pen.Redraw()
			elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
				app.pen.Continue(x, y)
				app.pen.Redraw()
			elif event == cv2.EVENT_LBUTTONUP:
				app.pen.End(x, y)
				app.pen.Redraw()

		cv2.setMouseCallback(self.imgName, mouseCallback, self)

		# self.pen.LoadTrack()
		# self.pen.Redraw()

		# 响应快捷键
		while True:
			font = cv2.FONT_HERSHEY_SIMPLEX
			msg = '[c]lear    [o]rigin'
			if self.conf.get('showOrigin'):
				msg += '-Y'

			msg += '    [l]ine origin'
			if self.conf.get('showOriginLine'):
				msg += '-Y'

			msg += '    AA[p]oly-line'
			if self.conf.get('showPolyLine'):
				msg += '-Y'

			msg += '     c[t]an'
			if self.conf.get('showCTan'):
				msg += '-Y'

			msg += '    [F2]save    [F5]load'
			cv2.putText(self.img, msg, (10, 490), font, 0.3, (0, 0, 0), 1)

			cv2.imshow(self.imgName, self.img)
			pressedKey = cv2.waitKey(10)	# 等待10ms，如果无按键，返回-1
			if pressedKey == -1:
				continue

			# logging.debug(pressedKey)
			needRedraw = False
			if pressedKey & 0xFF == 27:	# ESC 	退出
				break
			elif pressedKey & 0xFF == ord('c'):	# clean 	清屏
				needRedraw = True
			elif pressedKey & 0xFF == ord('o'):	# origin 	是否显示原始点 
				self.conf.set('showOrigin', (not self.conf.get('showOrigin')))
				needRedraw = True
			elif pressedKey & 0xFF == ord('l'): # line 		是否显示连接线
				self.conf.set('showOriginLine', (not self.conf.get('showOriginLine')))
				needRedraw = True
			elif pressedKey & 0xFF == ord('p'): # polyline 	显示抗锯齿曲线
				self.conf.set('showPolyLine', (not self.conf.get('showPolyLine')))
				needRedraw = True
			elif pressedKey & 0xFF == ord('t'): # ctan 		绘制法线
				self.conf.set('showCTan', (not self.conf.get('showCTan')))
				needRedraw = True
			elif pressedKey & 0xFF == 120: 		# F2 Save 	保存原始轨迹数据
				logging.debug('saving...')
				self.pen.SaveTrack()
				needRedraw = True
			elif pressedKey & 0xFF == 96: 		# F5 Load 	加载原始轨迹数据
				logging.debug('loading...')
				self.pen.LoadTrack()
				needRedraw = True

			if needRedraw:
				self.pen.Redraw()

		cv2.destroyAllWindows()

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    app = MagicPenApp()
    app.MainProc()