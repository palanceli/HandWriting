
# -*- coding:utf-8 -*-

import logging
import os
import numpy
import cv2
import scipy
import scipy.interpolate
import matplotlib
import matplotlib.pyplot

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
		logging.debug(pts)
		cv2.polylines(img, [pts], True, (255, 0, 0), 1)				# 多边形

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
		for i in range(200):
			img.itemset((10, i, 0), 255)
		for i in range(200):
			img.itemset((20, i, 1), 255)
		for i in range(200):
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
		img = cv2.copyMakeBorder(src, 10, 10, 10, 10, cv2.BORDER_REFLECT_101)
		img = cv2.copyMakeBorder(src, 10, 10, 10, 10, cv2.BORDER_WRAP)
		blue = (255, 0, 0)
		img = cv2.copyMakeBorder(src, 10, 10, 10, 10, cv2.BORDER_CONSTANT, blue) # 为什么画出来的不是蓝色？
		self.waitToClose(img)

	def case1001(self):
		# 算术运算
		x = numpy.uint8([250])
		y = numpy.uint8([10])
		logging.debug(cv2.add(x, y))	# opencv 是饱和操作 	250+10 => 255
		logging.debug(x + y)			# numpy  是模操作		250+10 => 4

	def case1002(self):
		# 图像混合，混合处理的公式是将两个图片每个像素做如下处理：
		# g(x) = (1-α)img1 + αimg2 + γ  α∈(0, 1)
		img1 = cv2.imread('sample01.jpg')
		height1, width1, nColor1 = img1.shape
		img2 = cv2.imread('sample02.png')
		img2 = cv2.resize(img2, (width1, height1)) 	# 图片拉伸，把img2拉成和img1同样大小
		dst = cv2.addWeighted(img1, 0.7, img2, 0.3, 0) # 其中α=0.3，γ=0
		self.waitToClose(dst)

	def caseR01(self):
		# 各种几何图形
		img = numpy.zeros((300, 300, 3), numpy.uint8)
		img[:, :] = (255, 255, 255)
		pts = numpy.array([[10, 10], [100, 10], [100, 100], [10, 100]], numpy.int32)
		pts = pts.reshape((-1, 1, 2))
		cv2.polylines(img, [pts], True, (0, 0, 0), 1)				# 多边形

		mask = numpy.zeros((302, 302, 0), numpy.uint8)
		mask[:] = 1

		cv2.floodFill(img, mask, (50, 50), (0, 0, 0))#, (50,)*3, (50,)*3, 4)

		self.waitToClose(img)

	def caseR02(self):
		# 填充多边形
		img = numpy.zeros((300, 300, 3), numpy.uint8)
		img[:, :] = (255, 255, 255)
		pts = numpy.array([[10, 10], [100, 10], [100, 100], [10, 100]], numpy.int32)
		pts = pts.reshape((-1, 1, 2))
		cv2.polylines(img, [pts], True, (0, 0, 0), 1)				# 多边形

		mask = numpy.zeros((302, 302), numpy.uint8)
		mask[:] = 0
		fixed_range = True
		connectivity = 4

		while True:
			ch = 0xFF & cv2.waitKey()
			if ch == 27:
				break
			if ch == ord('f'):
				fixed_range = not fixed_range #选定时flags的高位比特位0，也就是邻域的选定为当前像素与相邻像素的的差，这样的效果就是联通区域会很大
				print 'using %s range' % ('floating', 'fixed')[fixed_range]
			if ch == ord('c'):
				connectivity = 12-connectivity #选择4方向或则8方向种子扩散
				print 'connectivity =', connectivity

			flooded = img.copy()
			flags = connectivity
			if fixed_range:
				flags |= cv2.FLOODFILL_FIXED_RANGE 

			seedPoint = (50, 50)
			newVal = (0, 0, 0)
			loDiff = (0, )
			hiDiff = (0, )
			# flood 	源图
			# mask  	掩码图像，大小比原图多两个像素点。设输入图像大小为w * h ,
			#		 	则掩码的大小必须为 (w + 2) * (h + 2) , mask可为输出，也可作为输入 ，由flags决定。
			# seedPoint	起始填充标记点
			# newVal	新的填充值
			# loDiff 	为像素值的下限差值
			# hiDiff 	为像素值的上限差值
			# flags		0~7位为0x04或者0x08 即 4连通 或者 8连通，
			#			8~15位为填充mask的值大小 , 若为0 ， 则默认用1填充 
			# 			16~23位为 CV_FLOODFILL_FIXED_RANGE =(1 << 16), CV_FLOODFILL_MASK_ONLY =(1 << 17) 
			# 
			# 	flags参数通过位与运算处理：
			# 	当为CV_FLOODFILL_FIXED_RANGE 待处理的像素点与种子点作比较，如果∈(s – lodiff , s + updiff)，
			#			s为种子点像素值，则填充此像素。若无此位设置，则将待处理点与已填充的相邻点作此比较。
			# 	当为CV_FLOODFILL_MASK_ONLY 此位设置填充的对像， 若设置此位，则mask不能为空，
			# 			此时，函数不填充原始图像img，而是填充掩码图像. 若无此位设置，则在填充原始图像的时候，
			#			也用flags的8~15位标记对应位置的mask.
			cv2.floodFill(flooded, mask, seedPoint, newVal, loDiff, hiDiff, flags)

			cv2.circle(flooded, seedPoint, 2, (0, 0, 255), -1) #选定基准点用红色圆点标出
			cv2.imshow('floodfill', flooded)

		cv2.destroyAllWindows() 

	def caseR03(self):
		# 拟合曲线
		img = numpy.zeros((300, 300, 3), numpy.uint8)
		img[:, :] = (255, 255, 255)
		pts = numpy.array([[0, 0], [50, 50], [100, 50], [150, 0]], numpy.int32)
		pts = pts.reshape((-1, 1, 2))
		# print(pts)

		x = pts[:, 0][:, 0]
		y = pts[:, 0][:, 1]

		funcInterp = scipy.interpolate.interp1d(x, y, 'cubic')			# 生成样条插值函数

		xInterp = numpy.arange(1, 150, 1)
		yInterp = funcInterp(xInterp)
		logging.debug(yInterp)

		ptsInterp = numpy.zeros((len(xInterp), 2), numpy.float32)
		ptsInterp = ptsInterp.reshape((-1, 1, 2))

		ptsInterp[:, 0][:, 0] = xInterp
		ptsInterp[:, 0][:, 1] = yInterp

		cv2.polylines(img, numpy.int32([ptsInterp]), True, (255, 0, 0), 1)		# 多边形
		cv2.polylines(img, pts, True, (0, 0, 0), 2)		# 原始点

		mask = numpy.zeros((302, 302), numpy.uint8)
		mask[:] = 0
		fixed_range = True
		connectivity = 4

		flags = connectivity
		if fixed_range:
			flags |= cv2.FLOODFILL_FIXED_RANGE 

		seedPoint = (10, 5)
		newVal = (0, 0, 0)
		loDiff = (0, )
		hiDiff = (0, )
		cv2.floodFill(img, mask, seedPoint, newVal, loDiff, hiDiff, flags)

		self.waitToClose(img)

	def caseR06(self):
		img = numpy.zeros((350, 300, 3), numpy.uint8)
		img[:, :] = (255, 255, 255)
		pts = numpy.array([(240 , 200), (220, 320), (120, 200), (20, 320),], numpy.int32)
		pts = pts.reshape((-1, 1, 2))

		x = pts[:, 0][:, 0]
		y = pts[:, 0][:, 1]

		func = scipy.interpolate.interp1d(x, y, kind='linear')
		xInterp = numpy.linspace(x.min(), x.max(), 1000)
		yInterp = func(xInterp)

		ptsInterp[:, 0][:, 0] = xInterp
		ptsInterp[:, 0][:, 1] = yInterp
		cv2.polylines(img, numpy.int32([ptsInterp]), False, (255, 0, 0), 1)		# 多边形
		cv2.polylines(img, pts, True, (0, 0, 0), 2)		# 多边形
		cv2.polylines(img, numpy.int32(ptsInterp), True, (0, 0, 255), 2)		# 多边形

		logging.debug(pts)
		logging.debug(numpy.int32(ptsInterp))
		self.waitToClose(img)

	def caseR06_1(self):
		img = numpy.zeros((350, 300, 3), numpy.uint8)
		img[:, :] = (255, 255, 255)
		pts = numpy.array([(240 , 200), (220, 320), (120, 200), (20, 320),], numpy.int32)
		logging.debug(pts)
		pts = pts.reshape((-1, 1, 2))
		logging.debug(pts)

		cv2.polylines(img, [pts], False, (255, 0, 0), 1)		# 多边形

		self.waitToClose(img)

	def caseR07(self):
		logging.debug('生成[1, 8]的等差数列：')
		x = numpy.linspace(1, 8, 8)
		logging.debug(x)

		logging.debug('reshape成2*4的矩阵：')
		logging.debug(x.reshape((2, 4)))
		
		logging.debug('reshape成2*2*2的矩阵：')
		logging.debug(x.reshape((2, 2, 2)))

		logging.debug('reshape后的数组和原数组共享一段内存，如果原数组内容变化，变形后的一会变：')
		y = x.reshape(2, 4)
		y[1, 1] = 10
		logging.debug(y)
		logging.debug(x)

		logging.debug('reshape的值为-1，会根据数组长度和剩余维度中推断出来：')
		logging.debug(x.reshape(-1, 2, 2))



if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    s = samples()
    s.caseR07()
