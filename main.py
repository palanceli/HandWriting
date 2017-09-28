
# -*- coding:utf-8 -*-

import logging
import os
import numpy
import cv2
import math
import sys

class MPLine(object):
	# MPLine记录一条笔画，其中x y t是原始数据，由该类负责记录和保存，
	# extra是附加数据，也是和笔迹相关的数据，由调用方负责产生，MPLine只负责记录
	def __init__(self):
		# 每个元素是一个[x, y, t, extra] = MPPoint
		# x y t是基本值，extra是由笔迹效果根据基本值计算出来的
		self.data = []

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

	def AddBegin(self, x, y, t=None):
		newLine = MPLine()
		newLine.Add(x, y, t)
		self.trackList.append(newLine)
		return newLine

	def AddContinue(self, x, y, t=None):
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

	def AddEnd(self, x, y, t=None):
		return self.AddContinue(x, y, t)

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
	def Begin(self, x, y):
		return self.mpTracker.AddBegin(x, y)

	def Continue(self, x, y):
		return self.mpTracker.AddContinue(x, y)

	def End(self, x, y):
		return self.mpTracker.AddEnd(x, y)

	def Clean(self):
		self.mpTracker.Clean()
		self.img[:, :] = (255, 255, 255)

	def redrawSingleLine(self, mpLine, redrawAll=False):
		lastPoint = None
		polyLine = []
		dataLen = len(mpLine.data)
		for point in mpLine.data:
			# 只画最后几个点
			if redrawAll == False and dataLen - mpLine.data.index(point) > 3:
				continue
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

	def Redraw(self, redrawAll=False):
		return

	# SaveTrack、LoadTrack 保存和加载原始笔迹
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
	def __init__(self, lx, ly, rx, ry):
		self.data = {'lx':int(lx), 'ly':int(ly), 'rx':int(rx), 'ry':int(ry)}

	def GetLR(self):
		return self.data['lx'], self.data['ly'], self.data['rx'], self.data['ry'], 

class MagicPenBrush(MagicPen):
	def __init__(self, img, imgName, conf):
		MagicPen.__init__(self, img, imgName, conf)

	def Begin(self, x, y):
		MagicPen.Begin(self, x, y)

	def createExtraData(self, line, isEnd=False):
		if line is None or len(line.data) < 2:
			return

		pt0 = line.data[-1]
		x0, y0, t0 = pt0[0], pt0[1], pt0[2]
		pt1 = line.data[-2]
		x1, y1, t1 = pt1[0], pt1[1], pt1[2]
		width = 10
		distance = math.sqrt((y1 - y0)**2 + (x1 - x0)**2)
		wdconst = width / distance
		lx1 = x1 + (y0 - y1) * wdconst
		ly1 = y1 + (x1 - x0) * wdconst
		rx1 = x1 + (y1 - y0) * wdconst
		ry1 = y1 + (x0 - x1) * wdconst
		extraData = MPBrushExtra(lx1, ly1, rx1, ry1)
		pt1[3] = extraData
		return extraData

	def Continue(self, x, y):
		lastLine = MagicPen.Continue(self, x, y)
		extraData = self.createExtraData(lastLine, False)

	def End(self, x, y):
		lastLine = self.Continue(x, y)
		self.createExtraData(lastLine, True)

	def createCubic(self, x, y):		
		funcInterp = scipy.interpolate.interp1d(x, y, 'cubic')			# 生成样条插值函数

		xInterp = numpy.arange(1, 150, 1)
		yInterp = funcInterp(xInterp)

		ptsInterp = numpy.zeros((len(xInterp), 2), numpy.float32)
		ptsInterp = ptsInterp.reshape((-1, 1, 2))

	def getBSPLine(self, xList, yList):
		tck, u = scipy.interpolate.splprep([xList, yList], k=3, s=0)
		xInterp = numpy.linspace(0, 1, num=100, endpoint=True)
		out = scipy.interpolate.splev(xInterp, tck)

		ptsInterp = numpy.zeros((len(xInterp), 2), numpy.float32)
		ptsInterp = ptsInterp.reshape((-1, 1, 2))

		ptsInterp[:, 0][:, 0] = out[0]
		ptsInterp[:, 0][:, 1] = out[1]

	def redrawSingleLine(self, mpLine, redrawAll=False):
		MagicPen.redrawSingleLine(self, mpLine)

		dataLen = len(mpLine.data)
		for mpPoint in mpLine.data:
			# 只画最后几个点相关的数据
			currIndex = mpLine.data.index(mpPoint)
			if redrawAll == False and dataLen - currIndex > 4:
				continue

			extra = mpPoint[3]
			if extra == None:
				continue

			lx0, ly0, rx0, ry0 = extra.GetLR()

			# 绘制法线
			if self.conf.get('showCTan'):
				color = (0, 0, 0)
				cv2.line(self.img, (lx0, ly0), (rx0, ry0), color)

			if currIndex >= 4:
				lx1, ly1, rx1, ry1 = mpLine.data[currIndex - 1][3].GetLR()
				lx2, ly2, rx2, ry2 = mpLine.data[currIndex - 2][3].GetLR()
				lx3, ly3, rx3, ry3 = mpLine.data[currIndex - 3][3].GetLR()
				lx4, ly4, rx4, ry4 = mpLine.data[currIndex - 3][3].GetLR()


			lastExtra = lastPt[3]
			lx1, ly1 = lastExtra.data['lx'], lastExtra.data['ly']
			rx1, ry1 = lastExtra.data['rx'], lastExtra.data['ry']
			polyLine = [(lx0, ly0), (lx1, ly1)]
			polyLine = [numpy.array(polyLine, numpy.int32).reshape((-1, 1, 2))]
			color = (128, 128, 128)
			cv2.polylines(self.img, polyLine, True, color, 1)

	def Redraw(self, redrawAll=False):
		# 绘制每个点的法线段
		color = (0, 0, 0)
		for mpLine in self.mpTracker.trackList:
			if redrawAll:
				self.redrawSingleLine(mpLine, True)
			elif mpLine == self.mpTracker.trackList[-1]:
				self.redrawSingleLine(mpLine, False)

	def LoadTrack(self):
		MagicPen.LoadTrack(self)
		# load extra 
		for mpLine in self.mpTracker.trackList:
			lastPoint = None
			for mpPoint in mpLine.data:
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
				app.pen.Redraw(False)
			elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
				app.pen.Continue(x, y)
				app.pen.Redraw(False)
			elif event == cv2.EVENT_LBUTTONUP:
				app.pen.End(x, y)
				app.pen.Redraw(False)

		cv2.setMouseCallback(self.imgName, mouseCallback, self)

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
				self.pen.Clean()
				self.pen.Redraw(True)

		cv2.destroyAllWindows()

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    app = MagicPenApp()
    app.MainProc()