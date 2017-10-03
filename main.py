
# -*- coding:utf-8 -*-

import logging
import os
import numpy
import cv2
import math
import sys
import scipy
import scipy.interpolate

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
		self.data = {'lx':int(lx), 'ly':int(ly), 'rx':int(rx), 'ry':int(ry), 'width':width}

	def GetLR(self):
		return self.data['lx'], self.data['ly'], self.data['rx'], self.data['ry'], 

	def GetWidth(self):
		return self.data['width']

class LineWidthMgr(object):
	def GetWidth(self, vspeed):
		return 10

class LinearWidthMgr(LineWidthMgr):
	def GetWidth(self, vspeed):
		# 使用一次线性函数计算线宽
		maxWidth = 15
		minWidth = 5
		v1 = 4
		if vspeed > 4:
			return minWidth
		return (15 - (maxWidth - minWidth) * vspeed / v1)

class MagicPenBrush(MagicPen):
	def __init__(self, img, imgName, conf):
		MagicPen.__init__(self, img, imgName, conf)
		# 已经抬笔的笔画保存到backImg，每次重绘时使用self.img = self.backImg + 尚未抬笔的笔画
		self.backImg = self.img.copy()
		self.maskImg = numpy.zeros((self.backImg.shape[0] + 2, self.backImg.shape[1] + 2), numpy.uint8)
		self.maskImg[:] = 0
		self.widthMgr = LinearWidthMgr()

	def Clean(self):
		MagicPen.Clean(self)
		self.backImg[:, :] = (255, 255, 255)

	def Begin(self, x, y, t=None):
		MagicPen.Begin(self, x, y, t)

	def createPointExtraData(self, line, isEnd=False):
		if line is None or len(line.data) < 2:
			return

		pt0 = line.data[-1]
		x0, y0, t0 = pt0[0], pt0[1], pt0[2]
		pt1 = line.data[-2]
		x1, y1, t1 = pt1[0], pt1[1], pt1[2]

		# width2 = 0
		# if len(line.data) == 2:
		# 	width2 = pt1[3].GetWidth()
		# else:
		# 	pt2 = line.data[-3]
		# 	width2 = pt2[3].GetWidth()

		distance = math.sqrt((y1 - y0)**2 + (x1 - x0)**2)
		VSpeed = distance * 1000000 / (t0 - t1)
		width = self.widthMgr.GetWidth(VSpeed)
		# if len(line.data) > 2:
		# 	pt2 = line.data[-3]
		# 	width2 = pt2[3].GetWidth()
		# 	if abs(width - width2) / (2 * distance) > (1.732 / 3):
		# 		width = width2 + 2 * distance / 1.732
		# 	logging.debug(abs(width - width2) / distance)

		logging.debug('%12d    %4.1f    %4.4f    %2.1f' % (t0-t1, distance, VSpeed, width))
		wdconst = width / distance
		lx1 = x1 + (y0 - y1) * wdconst
		ly1 = y1 + (x1 - x0) * wdconst
		rx1 = x1 + (y1 - y0) * wdconst
		ry1 = y1 + (x0 - x1) * wdconst
		extraData = MPBrushPointExtra(lx1, ly1, rx1, ry1, width)
		pt1[3] = extraData
		return extraData

	def createCubicInterp(self, x, y):
		tck, u = scipy.interpolate.splprep([x,y], k=3, s=0)
		xInterp = numpy.linspace(0, 1, num=100, endpoint=True)
		out = scipy.interpolate.splev(xInterp, tck)
		xInterps = out[0]
		yInterps = out[1]
		return xInterps, yInterps

	def createOutline(self, line):
		lxList = []
		lyList = []
		rxList = []
		ryList = []
		if line is None:
			return None

		for pt in line.data:
			ptExtra = pt[3]
			if ptExtra is None:
				continue
			lx, ly, rx, ry = ptExtra.GetLR()
			lxList.append(lx)
			lyList.append(ly)
			rxList.append(rx)
			ryList.append(ry)
		if len(lxList) < 4 or len(rxList) < 4:
			return None
		# logging.debug(lxList)
		# logging.debug(lyList)
		lxInterps, lyInterps = self.createCubicInterp(lxList, lyList) 	# 左边界插值
		rxInterps, ryInterps = self.createCubicInterp(rxList, ryList)	# 右边界插值
		
		# 生成封闭轮廓
		outline = numpy.zeros((len(lxInterps) + len(rxInterps) + 1, 2), numpy.float32)

		outline[ : len(lxInterps), 0] = lxInterps
		outline[ : len(lyInterps), 1] = lyInterps
		outline[len(lxInterps), 0] = line.data[-1][0]
		outline[len(lyInterps), 1] = line.data[-1][1]
		outline[len(lxInterps) + 1 : , 0] = rxInterps[::-1]
		outline[len(lyInterps) + 1 : , 1] = ryInterps[::-1]

		line.extra = outline
		return line.extra

	def Continue(self, x, y, t=None):
		lastLine = MagicPen.Continue(self, x, y, t) 	# 记录原始笔迹
		pointExtraData = self.createPointExtraData(lastLine, False)	# 为每个笔记点生成左右点
		lineExtraData = self.createOutline(lastLine)	# 插值
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
		fillColor = (100, 100, 100)
		outlinePts = numpy.int32(mpLine.extra)
		# logging.debug(outlinePts)
		cv2.polylines(img, [outlinePts], True, fillColor, 1, cv2.LINE_AA)
		cv2.fillPoly(img, [outlinePts], fillColor)
		# cv2.fillConvexPoly(img, outlinePts, fillColor)
		return
		# polyPts = numpy.zeros((len(mpLine.data), 2), numpy.int32)
		# polyPts[:, 0] = [pt[0] for pt in mpLine.data]
		# polyPts[:, 1] = [pt[1] for pt in mpLine.data]
		# polyPts = polyPts.reshape(-1, 1, 2)
		# # logging.debug(polyPts)
		# # cv2.polylines(img, polyPts, True, blackColor, 3) 		# 绘制原始点
		# # cv2.polylines(img, [polyPts], False, lightColor, 1) 	# 绘制原始笔迹

		# lPts = []
		# rPts = []
		# # 法线左右两点
		# for mpPoint in mpLine.data:
		# 	ptExtra = mpPoint[3]
		# 	if ptExtra == None:
		# 		continue

		# 	lx0, ly0, rx0, ry0 = ptExtra.GetLR()
		# 	# cv2.line(img, (lx0, ly0), (rx0, ry0), blackColor)	# 绘制法线
		# 	lPts.append((lx0, ly0))
		# 	rPts.append((rx0, ry0))

		# if len(lPts) > 2 and len(rPts) > 2:
		# 	# 填充
		# 	last1Pt = mpLine.data[-1]
		# 	x, y = last1Pt[0], last1Pt[1]

		# 	borderPts = lPts
		# 	borderPts.append((x, y))
		# 	borderPts.extend(rPts[::-1])
		# 	borderPts = numpy.array([borderPts], numpy.int32).reshape(-1, 1, 2)
		# 	cv2.polylines(img, [borderPts], True, fillColor, 1, cv2.LINE_AA)
		# 	cv2.fillPoly(img, [borderPts], fillColor)

	def End(self, x, y, t=None):
		lastLine = self.Continue(x, y, t)
		# self.createLineExtraData(lastLine)
		# 抬笔的时候把最后一笔画到backImg上
		self.drawMPLineToImg(lastLine, self.backImg)
		logging.debug('抬笔')

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

		self.pen.LoadTrack()
		self.pen.Redraw()

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