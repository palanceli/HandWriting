
# -*- coding:utf-8 -*-

import logging
import os
import numpy
import cv2

class samples(object):
	def waitToClose(self, img):
		while True:
			cv2.imshow('image', img)
			if cv2.waitKey(20) & 0xFF == 27:
				break
		
		cv2.destroyAllWindows()

	def case0101(self):
		# 读入一张图片
	    img = cv2.imread('sample01.jpg', 0)
	    self.waitToClose(img)

	def case0601(self):
		# 各种几何图形
		img = numpy.zeros((512, 512, 3), numpy.uint8)
		cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 2)			# 直线
		cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 2)	# 矩形
		cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)				# 圆
		cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1)	# 椭圆

		pts = numpy.array([[10, 5], [20, 30], [70, 20], [50, 10]], numpy.int32)
		pts = pts.reshape((-1, 1, 2))
		cv2.polylines(img, pts, True, (255, 0, 0), 1)				# 多边形

		font = cv2.FONT_HERSHEY_SIMPLEX								# 文字
		cv2.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2)

		self.waitToClose(img)

	def case0701(self):
		# 在鼠标双击的地方绘制圆圈
		events = [i for i in dir(cv2) if 'EVENT' in i]
		logging.debug(events)

		def drawCircle(event, x, y, flags, param):
			if event == cv2.EVENT_LBUTTONDBLCLK:
				logging.debug('(%d, %d)' % (x, y))
				img = param
				cv2.circle(img, (x, y), 10, (255, 0, 0), 1)

		img = numpy.zeros((512, 512, 3), numpy.uint8)
		cv2.namedWindow('image')
		cv2.setMouseCallback('image', drawCircle, img)
		self.waitToClose(img)

	def case0702(self):
		# 形状随鼠标拖动而变化
		class DrawGraphParam(object):
			def __init__(self, img):
				self.img = img
				self.bDrawing = False	# 鼠标按下为True，抬起为False
				self.mode = True 		# 为True绘制矩形，按下m为False，绘制圆
				self.ix, self.iy = -1, -1

		def drawGraph(event, x, y, flags, param):
			dgp = param
			if event == cv2.EVENT_LBUTTONDOWN:
				# 按下左键记录起始位置
				dgp.bDrawing = True
				dgp.ix, dgp.iy = x, y
				logging.debug('start')
			elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
				logging.debug('drawing...')
				# 按住左键移动，绘制图形
				if dgp.bDrawing == True:
					logging.debug('bDrawing == True')
					if dgp.mode == True:
						logging.debug('draw rectangle')
						cv2.rectangle(dgp.img, (dgp.ix, dgp.iy), (x, y), (0, 255, 0), 1)
					else:
						logging.debug('draw circle')
						r = int(numpy.sqrt((x - dgp.ix) ** 2 + (y - dgp.iy) ** 2))
						cv2.circle(dgp.img, (dgp.ix, dgp.iy), r, (0, 0, 255), 1)
			else:
				# 鼠标松开停止绘制
				dgp.bDrawing = False

		img = numpy.zeros((512, 512, 3), numpy.uint8)
		dgp = DrawGraphParam(img)
		cv2.namedWindow('image')
		cv2.setMouseCallback('image', drawGraph, dgp)
		while True:
			cv2.imshow('image', dgp.img)
			k = cv2.waitKey(1) & 0xFF
			if k == ord('m'):
				dgp.mode = not dgp.mode
			elif k == 27:
				break

	def case0801(self):
		# 在界面上添加几个滑块，用来控制背景色
		def nothing(x):
			pass

		img = numpy.zeros((300, 512, 3), numpy.uint8)
		img[:] = 255
		cv2.namedWindow('image')

		# 参数分别表示：滚动条名字，滚动条所在窗口的名字，默认位置，最大值，回调
		cv2.createTrackbar('R', 'image', 0, 255, nothing)
		cv2.createTrackbar('G', 'image', 0, 255, nothing)
		cv2.createTrackbar('B', 'image', 0, 255, nothing)

		cv2.createTrackbar('0:0FF\n1:ON', 'image', 0, 1, nothing)

		while True:
			cv2.imshow('image', img)

			k = cv2.waitKey(1) & 0xFF
			if k == 27:
				break

			r = cv2.getTrackbarPos('R', 'image')
			g = cv2.getTrackbarPos('G', 'image')
			b = cv2.getTrackbarPos('B', 'image')
			s = cv2.getTrackbarPos('0:0FF\n1:ON', 'image')

			if s == 0:
				img[:] = 0
			else:
				img[:] = [b, g, r]
		cv2.destroyAllWindows()

	def case0901(self):
		img = cv2.imread('sample01.jpg')
		px = img[100, 100]	# (b,g,r) = image[i,j]，image大小为：MxNxK
		logging.debug(px)
		img[100:110, 100:110] = [255, 0, 0]

		# img.item(x, y, iRGB) iRGB：0 - B, 1 - G, 2 - R
		logging.debug(img.item(10, 10, 2))
		for i in range(300):
			img.itemset((10, i, 0), 255)
		for i in range(300):
			img.itemset((20, i, 1), 255)
		for i in range(300):
			img.itemset((30, i, 2), 255)

		logging.debug(img.shape)	# (rows, cols, type)
		# type表示矩阵中元素的类型以及矩阵的通道个数。其命名规则为CV_(位数）+（数据类型）+（通道数）
		# 例如：CV_16UC2，表示的是元素类型是一个16位的无符号整数，通道为2
		msg = ''
		c = 0
		for i in dir(cv2):
			if 'CV_' not in i:
				continue
			c += 1
			if c % 5 == 0:
				msg += '\n'
			msg += '%9s:%2d ' % (i, getattr(cv2, i))
		logging.debug(msg)

		logging.debug(img.size)	# 像素的个数
		logging.debug(img.dtype)	# 返回图像的数据类型

		b, g, r = cv2.split(img)	# 图像的拆分
		img = cv2.merge(b, g, r)	# 图像的合并

		img[:, :, 2] = 0			# 将图像中的红色通道的值设为0

		self.waitToClose(img)

	def case0905(self):
		# 为源图扩边
		src = cv2.imread('sample01.jpg')
		# 参数含义
		# 输入图像，top, bottom, left, right, BorderType, 边界颜色
		# 	BorderType的类型：
		#	cv2.BORDER_CONSTANT 添加有颜色的常数边界，还需要下一个参数value
		#	cv2.BORDER_REFLECT 边界元素的镜像。比如: fedcba|abcdefgh|hgfedcb
		#	cv2.BORDER_REFLECT_101 or cv2.BORDER_DEFAULT跟上面一样，但稍作改动。例如: gfedcb|abcdefgh|gfedcba
		#	cv2.BORDER_REPLICATE 重复最后一个元素。例如: aaaaaa|abcdefgh|hhhhhhh
		#	cv2.BORDER_WRAP 不知道怎么说了, 就像这样: cdefgh|abcdefgh|abcdefg
		img = cv2.copyMakeBorder(src, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
		img = cv2.copyMakeBorder(src, 10, 10, 10, 10, cv2.BORDER_REFLECT)
		self.waitToClose(img)


if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    s = samples()
    s.case0905()