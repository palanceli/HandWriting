
# -*- coding:utf-8 -*-

import logging
import os
import numpy
import cv2

class HandWriting(object):
	def __init__(self):
		# 这是一个笔迹数组，每个元素都是[(x0, y0, t0), (x1, y1, t1), ...]，表示一条原始笔迹
		self.originTrackList = [] 
		self.img = numpy.zeros((500, 899, 3), numpy.uint8)
		self.img[:, :] = (255, 255, 255)
		self.imgName = 'image'

	def drawX(self, x, y):
		x0 = x - 2
		y0 = y - 2
		x1 = x + 2
		y1 = y + 2
		cv2.line(self.img, (x0, y0), (x1, y1), (0, 0, 0), 1)
		x0 = x + 2
		y0 = y - 2
		x1 = x - 2
		y1 = y + 2
		cv2.line(self.img, (x0, y0), (x1, y1), (0, 0, 0), 1)

	def beginTrack(self, x, y):
		point = (x, y, cv2.getTickCount())
		logging.debug('beginTrack:(%d, %d, %d)' % point)
		self.originTrackList.append([point])
		self.drawX(x, y)

	def continueTrack(self, x, y):
		point = (x, y, cv2.getTickCount())
		lastestTrack = self.originTrackList[-1]
		lastestTrack.append(point)
		self.drawX(x, y)

	def endTrack(self, x, y):
		point = (x, y, cv2.getTickCount())
		lastestTrack = self.originTrackList[-1]
		lastestTrack.append(point)
		self.drawX(x, y)

	def MainProc(self):
		# 读入一张图片
		cv2.imshow(self.imgName, self.img)
		cv2.moveWindow(self.imgName, 100, 100)

		def mouseCallback(event, x, y, flags, param):
			hw = param
			if event == cv2.EVENT_LBUTTONDOWN:
				hw.beginTrack(x, y)
			elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
				hw.continueTrack(x, y)
			elif event == cv2.EVENT_LBUTTONUP:
				hw.endTrack(x, y)

		cv2.setMouseCallback('image', mouseCallback, self)
		
		while True:
			cv2.imshow('image', self.img)
			if cv2.waitKey(20) & 0xFF == 27:
				break
		
		cv2.destroyAllWindows()

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    hw = HandWriting()
    hw.MainProc()