
# -*- coding:utf-8 -*-

import logging
import os
import numpy
import cv2
import math
import sys

class MPLine(object):
	# 这是一条笔迹线
	def __init__(self):
		# 每个元素是一个[x, y, t, extra] = MPPoint
		# x y t是基本值，extra是由笔迹效果根据基本值计算出来的
		self.line = []

	def Add(self, x, y, t=None):
		if t == None:
			t = cv2.getTickCount()
		point = [x, y, t, None]
		self.line.append(point)
		return point

	def ToString(self):
		lineString = ''
		for point in self.line:
			x, y, t = point[0], point[1], point[2]
			lineString += '[%d,%d,%d]' % (x, y, t)
		return lineString

	def FromString(self, lineString):
		pointList = lineString.strip().split('][')
		for point in pointList:
			x, y, t = point.strip('[').strip(']').split(',')
			self.Add(int(x), int(y), int(t))

	def GetPointNum(self):
		return len(self.line)

class MPTracker(object):
	def __init__(self):
		# 这是一个笔迹数组，每个元素都是MPLine
		self.trackList = []
		self.filePath = 'originTracker.data'

	def AddBegin(self, x, y, t=None):
		newLine = MPLine()
		newLine.Add(x, y, t)
		self.trackList.append(newLine)
		return newLine

	def AddContinue(self, x, y, t=None):
		if len(self.trackList) == 0:
			return self.BeginTrack(x, y, t)
		lastLine = self.trackList[-1]
		# 如果x,y和前一个点坐标一致，就不再加了
		if lastLine.GetPointNum() > 0:
			lastPoint = lastLine.line[-1]
			if lastPoint[0] == x and lastPoint[1] == y:
				return lastLine
		lastLine.Add(x, y, t)
		return lastLine

	def AddEnd(self, x, y, t=None):
		return self.ContinueTrack(x, y, t)

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
				if len(newLine.line) > 0:
					self.trackList.append(newLine)

class MagicPenConf(object):
	def __init__(self):
		self.conf = {'showOrigin' : True, 'showOriginLine' : False, 'showPolyLine' : True,
		'showCTan':False}

	def set(self, key, value):
		self.conf[key] = value

	def get(self, key):
		return self.conf[key]

class MagicPen(object):
	def __init__(self, img, imgName, conf):
		self.mpTracker = MPTracker()
		self.img = img
		self.imgName = imgName
		self.conf = conf

	def Begin(self, x, y):
		return self.mpTracker.AddBegin(x, y)

	def Continue(self, x, y):
		return self.mpTracker.AddContinue(x, y)

	def End(self, x, y):
		return self.mpTracker.AddEnd(x, y)

	def Clean(self):
		self.mpTracker.Clean()
		self.img[:, :] = (255, 255, 255)

	def Redraw(self):
		self.img[:, :] = (255, 255, 255)
		for mpLine in self.mpTracker.trackList:
			lastPoint = None
			polyLine = []
			for point in mpLine.line:
				x, y, t = point[0], point[1], point[2]
				self.drawX(x, y)
				polyLine.append((x, y))

				# 和前一个点连成线
				if lastPoint == None:
					lastPoint = point
					continue
				x0, y0, t0 = lastPoint[0], lastPoint[1], lastPoint[2]
				self.drawOriginLine(x0, y0, x, y)
				lastPoint = point

			if self.conf.get('showPolyLine'):
				polyLine = [numpy.array(polyLine, numpy.int32).reshape((-1, 1, 2))]
				color = (128, 128, 128)
				cv2.polylines(self.img, polyLine, False, color, 1, cv2.LINE_AA)

	def SaveTrack(self):
		self.mpTracker.Save()

	def LoadTrack(self):
		self.mpTracker.Load()

	def drawX(self, x, y):
		if not self.conf.get('showOrigin'):
			return
		color = (224, 224, 224)
		x0, y0, x1, y1 = x - 1, y - 1, x + 1, y + 1
		cv2.line(self.img, (x0, y0), (x1, y1), color, 1)
		x0, y0, x1, y1 = x + 1, y - 1, x - 1, y + 1
		cv2.line(self.img, (x0, y0), (x1, y1), color, 1)

	def drawOriginLine(self, x0, y0, x1, y1):
		if not self.conf.get('showOriginLine'):
			return
		color = (0, 0, 0)
		cv2.line(self.img, (x0, y0), (x1, y1), color)

class MPBrushExtra(object):
	# 附着在每个MPPoint 的extra数据
	def __init__(self, x0, y0, t0, x1, y1, t1):
		self.lx, self.ly = 0, 0
		self.rx, self.ry = 0, 0
		self.calc(x0, y0, t0, x1, y1, t1)

	def calc(self, x0, y0, t0, x1, y1, t1):
		# 根据前后两个点计算(x0, y0)法线上的左右两点
		width = 10
		distance = math.sqrt((y1 - y0)**2 + (x1 - x0)**2)
		v = distance * 1000000 / (t1 - t0)
		# logging.debug('%d / %d = %.2f' % (distance * 1000000, t1 - t0, v))
		logging.debug(cv2.log(v)[0][0])
		width = width / (1 + cv2.exp(cv2.log(v)[0][0]))[0][0]
		logging.debug(width)
		# width += v[0][0]
		# logging.debug('(%d, %d) - (%d, %d) = %d' % (x1, y1, x0, y0, distance))
		if distance == 0:
			sys.exit(0)
		deltaX = (y0 - y1) * width / distance
		deltaY = (x1 - x0) * width / distance
		self.lx, self.ly = int(x0 - deltaX), int(y0 - deltaY)
		self.rx, self.ry = int(x0 + deltaX), int(y0 + deltaY)

class MagicPenBrush(MagicPen):
	def __init__(self, img, imgName, conf):
		MagicPen.__init__(self, img, imgName, conf)

	def Begin(self, x, y):
		MagicPen.Begin(self, x, y)
		self.drawX(x, y)

	def drawCTan(self, extra):
		if not self.conf.get('showCTan'):
			return
		color = (0, 0, 0)
		cv2.line(self.img, (extra.lx, extra.ly), (extra.rx, extra.ry), color)

	def Continue(self, x, y):
		lastLine = MagicPen.Continue(self, x, y)
		self.drawX(x, y)	# 绘制点

		# 当前点和前一个点连成线
		if lastLine.GetPointNum() <= 1: # 如果没有前一个点则返回
			return
		lastPoint = lastLine.line[-2]
		x0, y0, t0 = lastPoint[0], lastPoint[1], lastPoint[2]
		currPoint = lastLine.line[-1]
		x1, y1, t1 = currPoint[0], currPoint[1], currPoint[2]
		self.drawOriginLine(x0, y0, x1, y1)

		# 根据前后两点计算前面一个点的extra数据
		extra = MPBrushExtra(x0, y0, t0, x1, y1, t1)
		lastPoint[3] = extra
		self.drawCTan(extra)

	def End(self, x, y):
		self.Continue(x, y)

	def Redraw(self):
		MagicPen.Redraw(self)

		# 绘制每个点的法线段
		color = (0, 0, 0)
		for mpLine in self.mpTracker.trackList:
			lastExtra = None
			for mpPoint in mpLine.line:
				extra = mpPoint[3]
				if extra == None:
					continue
				self.drawCTan(extra)

				if lastExtra == None:
					lastExtra = extra
					continue

				polyLine = [(lastExtra.lx, lastExtra.ly), (extra.lx, extra.ly), 
				(extra.rx, extra.ry), (lastExtra.rx, lastExtra.ry)]
				polyLine = [numpy.array(polyLine, numpy.int32).reshape((-1, 1, 2))]
				color = (128, 128, 128)
				cv2.polylines(self.img, polyLine, True, color, 1)

				lastExtra = extra

	def LoadTrack(self):
		MagicPen.LoadTrack(self)
		# load extra 
		for mpLine in self.mpTracker.trackList:
			lastPoint = None
			for mpPoint in mpLine.line:
				if lastPoint == None:
					lastPoint = mpPoint
					continue
				x0, y0, t0 = lastPoint[0], lastPoint[1], lastPoint[2]
				x1, y1, t1 = mpPoint[0], mpPoint[1], mpPoint[2]
				extra = MPBrushExtra(x0, y0, t0, x1, y1, t1)
				lastPoint[3] = extra
				lastPoint = mpPoint

class MagicPenApp(object):
	def __init__(self):
		self.img = self.createImg()
		self.imgName = 'image'
		self.conf = MagicPenConf()
		self.pen = MagicPenBrush(self.img, self.imgName, self.conf)

	def createImg(self):
		img = numpy.zeros((500, 899, 3), numpy.uint8)
		img[:, :] = (255, 255, 255)
		return img

	def MainProc(self):
		cv2.imshow(self.imgName, self.img)
		cv2.moveWindow(self.imgName, 100, 100)

		def mouseCallback(event, x, y, flags, param):
			app = param
			if event == cv2.EVENT_LBUTTONDOWN:
				app.pen.Begin(x, y)
			elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
				app.pen.Continue(x, y)
			elif event == cv2.EVENT_LBUTTONUP:
				app.pen.End(x, y)

		cv2.setMouseCallback(self.imgName, mouseCallback, self)
		
		while True:
			cv2.imshow(self.imgName, self.img)
			pressedKey = cv2.waitKey(10)	# 等待10ms，如果无按键，返回-1
			if pressedKey == -1:
				continue

			# logging.debug(pressedKey)
			if pressedKey & 0xFF == 27:	# ESC 	退出
				break
			elif pressedKey & 0xFF == ord('c'):	# clean 	清屏
				self.pen.Clean()
			elif pressedKey & 0xFF == ord('o'):	# origin 	是否显示原始点 
				self.conf.set('showOrigin', (not self.conf.get('showOrigin')))
				self.pen.Redraw()
			elif pressedKey & 0xFF == ord('l'): # line 		是否显示连接线
				self.conf.set('showOriginLine', (not self.conf.get('showOriginLine')))
				self.pen.Redraw()
			elif pressedKey & 0xFF == ord('p'): # polyline 	显示抗锯齿曲线
				self.conf.set('showPolyLine', (not self.conf.get('showPolyLine')))
				self.pen.Redraw()
			elif pressedKey & 0xFF == ord('t'): # ctan 		绘制法线
				self.conf.set('showCTan', (not self.conf.get('showCTan')))
				self.pen.Redraw()
			elif pressedKey & 0xFF == 120: 		# F2 Save 		保存原始轨迹数据
				logging.debug('saving...')
				self.pen.SaveTrack()
			elif pressedKey & 0xFF == 96: 		# F5 Load 		加载原始轨迹数据
				logging.debug('loading...')
				self.pen.LoadTrack()
				self.pen.Redraw()

		cv2.destroyAllWindows()

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    app = MagicPenApp()
    app.MainProc()