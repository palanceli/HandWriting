
# -*- coding:utf-8 -*-

import logging
import os
import numpy
import cv2

class OriginTracker(object):
	def __init__(self):
		# 这是一个笔迹数组，每个元素都是[(x0, y0, t0), (x1, y1, t1), ...]，表示一条原始笔迹
		self.trackList = []
		self.filePath = 'originTracker.data'

	def AddBegin(self, x, y, t=None):
		if t == None:
			t = cv2.getTickCount()
		point = (x, y, t)
		newTrack = [point]
		self.trackList.append(newTrack)
		return newTrack

	def AddContinue(self, x, y, t=None):
		if t == None:
			t = cv2.getTickCount()

		if len(self.trackList) == 0:
			return self.BeginTrack(x, y, t)
		
		point = (x, y, t)
		lastestTrack = self.trackList[-1]
		lastestTrack.append(point)
		return lastestTrack

	def AddEnd(self, x, y, t=None):
		return self.ContinueTrack(x, y, t)

	def Clean(self):
		self.trackList = []

	def Save(self):
		with open(self.filePath, 'wb') as f:
			for track in self.trackList:
				for point in track:
					f.write('{%d,%d,%d}' % point)
				f.write('\n')

	def Load(self):
		self.Clean()
		with open(self.filePath, 'rb') as f:
			for line in f:
				pointList = line.strip().split('}{')
				count = 0
				for point in pointList:
					(x, y, t) = point.strip('}').strip('{').split(',')
					x = int(x)
					y = int(y)
					t = int(t)
					if count == 0:
						self.AddBegin(x, y, t)
					else:
						self.AddContinue(x, y, t)
					count += 1

class MagicPenConf(object):
	def __init__(self):
		self.conf = {'showOrigin' : True, 'showOriginLine' : False, 'showPolyLine' : True}

	def set(self, key, value):
		self.conf[key] = value

	def get(self, key):
		return self.conf[key]

class MagicPen(object):
	def __init__(self, img, imgName, conf):
		self.originTracker = OriginTracker()
		self.img = img
		self.imgName = imgName
		self.conf = conf

	def Begin(self, x, y):
		return self.originTracker.AddBegin(x, y)

	def Continue(self, x, y):
		return self.originTracker.AddContinue(x, y)

	def End(self, x, y):
		return self.originTracker.AddEnd(x, y)

	def Clean(self):
		self.originTracker.Clean()
		self.img[:, :] = (255, 255, 255)

	def Redraw(self):
		self.img[:, :] = (255, 255, 255)
		for track in self.originTracker.trackList:
			lastPoint = None
			line = []
			for point in track:
				self.drawX(point[0], point[1])
				if lastPoint == None:
					lastPoint = point
					continue
				self.drawOriginLine(lastPoint[0], lastPoint[1], point[0], point[1])
				lastPoint = point
				line.append((point[0], point[1]))
			if self.conf.get('showPolyLine'):
				cv2.polylines(self.img, [numpy.array(line, numpy.int32).reshape((-1, 1, 2))], False, (128, 128, 128), 1, cv2.LINE_AA)

	def SaveOriginTrack(self):
		self.originTracker.Save()

	def LoadOriginTrack(self):
		self.originTracker.Load()

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

class MagicPenBrush(MagicPen):
	def __init__(self, img, imgName, conf):
		MagicPen.__init__(self, img, imgName, conf)

	def Begin(self, x, y):
		MagicPen.Begin(self, x, y)
		self.drawX(x, y)

	def Continue(self, x, y):
		lastestTrack = MagicPen.Continue(self, x, y)
		self.drawX(x, y)
		if len(lastestTrack) <= 1:
			return
		lastPoint = lastestTrack[-2]
		self.drawOriginLine(lastPoint[0], lastPoint[1], x, y)

	def End(self, x, y):
		self.Continue(x, y)

	def Redraw(self):
		MagicPen.Redraw(self)

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

			logging.debug(pressedKey)
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
			elif pressedKey & 0xFF == 102: 		# F2 Save 		保存原始轨迹数据
				self.pen.SaveOriginTrack()
			elif pressedKey & 0xFF == 96: 		# F5 Load 		加载原始轨迹数据
				self.pen.LoadOriginTrack()
				self.pen.Redraw()

		cv2.destroyAllWindows()

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    app = MagicPenApp()
    app.MainProc()