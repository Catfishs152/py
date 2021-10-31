# coding=utf-8
import numpy as np
import cv2 as cv

drawing = False #按下鼠标则为真
savepath = 'C:/Users/11037/Desktop/12.jpg' #图片保存位置
R = 0
G = 0
B = 0
def nothing(x):
    pass

def draw(event,x,y,flags,param):
	global drawing,R,G,B
	if event == cv.EVENT_LBUTTONDOWN:  #响应鼠标按下
		drawing = True
	elif event == cv.EVENT_MOUSEMOVE: #响应鼠标移动
		if drawing == True:
			img[y:y+5,x:x+5] = (R,G,B)
	elif event == cv.EVENT_LBUTTONUP:  #响应鼠标松开
		drawing = False

# 创建一个黑色的图像，一个窗口
img = np.zeros((300,300,3), np.uint8)
cv.namedWindow('image')
# 创建颜色变化的轨迹栏
save = 'Save'
clear = 'clear'

cv.createTrackbar(clear,'image',0,1,nothing)
cv.createTrackbar(save, 'image',0,1,nothing)
cv.createTrackbar('R','image',0,255,nothing)
cv.createTrackbar('G','image',0,255,nothing)
cv.createTrackbar('B','image',0,255,nothing)
cv.setMouseCallback('image',draw)
img[:] = (255,255,255) #将画板设为白色
while(1):
	img[0:20,0:20] = (R,G,B)
	cv.imshow('image',img)
	if cv.waitKey(1)&0xFF == 27:
		break
	s = cv.getTrackbarPos(save,'image')
	c = cv.getTrackbarPos(clear,'image')
	R = cv.getTrackbarPos('R', 'image')
	G = cv.getTrackbarPos('G','image')
	B = cv.getTrackbarPos('B','image')
	#保存图片
	if s == 1:
		cv.imwrite(savepath,img)
		cv.setTrackbarPos(save,'image',0)
		img[:] = (255,255,255)
	#清空画布
	if c == 1:
		cv.setTrackbarPos(clear, 'image', 0)
		img[:] = (255, 255, 255)

cv.destroyAllWindows()
