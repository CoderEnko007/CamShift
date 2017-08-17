import numpy as np
import argparse
import cv2

frame = None
roiPts = []
inputMode = False

def selectROI(event, x, y, flag, param):
	global frame, roiPts, inputMode
	if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
		roiPts.append((x, y))
		cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
		cv2.imshow("frame", frame)

def main():
	global frame, roiPts, inputMode
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video")
	args = vars(ap.parse_args())

	if not args.get("video", False):
		camera = cv2.VideoCapture(0)
	else:
		camera = cv2.VideoCapture(args["video"])

	cv2.namedWindow("frame")
	cv2.setMouseCallback("frame", selectROI)

	termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
	roiBox = None

	while True:
		(grabed, frame) = camera.read()
		
		if not grabed:
			break
		if roiBox is not None:
			hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
			'''
			[0]表示通道列表，这里表示只用Hue通道进行匹配， 
			[0, 180]每一维直方图bin的范围，
			1表示反向投影的比例
			backProj表示各点匹配roiHist的概率，在meanShift算法中可以理解为各个离散的点，密度最高的点即匹配模板的区域
			此步骤获取反向投影，其中的值为相应位置元素匹配模板的概率，结果为灰度图，越白表示概率越高
			'''
			backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)
			cv2.imshow("backProj", backProj)
			print("roiBox1="+str(roiBox))
			'''
			CamShift每次会重新获取新的roi存入roiBox
			r包含对象位置，大小和方向的旋转矩形结构。
			'''
			(r, roiBox) = cv2.CamShift(backProj, roiBox, termination)
			#print("r="+str(r))
			print("roiBox="+str(roiBox))
			pts = np.int0(cv2.boxPoints(r))
			cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
		cv2.imshow("frame", frame)
		key = cv2.waitKey(1) & 0xFF

		if key == ord("i") and len(roiPts) < 4:
			inputMode = True
			orig = frame.copy()

			while len(roiPts) < 4:
				cv2.imshow("frame", frame)
				cv2.waitKey(0)

			roiPts = np.array(roiPts)
			s = roiPts.sum(axis = 1)
			tl = roiPts[np.argmin(s)]
			br = roiPts[np.argmax(s)]

			roi = orig[tl[1]:br[1], tl[0]:br[0]]
			roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

			roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
			cv2.imshow("roiHist1", roiHist)
			roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
			cv2.imshow("roiHist2", roiHist)
			file = open("test.txt", "w")
			file.write(str(roiHist))
			file.close()
			roiBox = (tl[0], tl[1], br[0], br[1])
		elif key == ord("q"):
			break
	camera.release()
	cv2.destroyAllWindows()

'''
__name__系统变量指示模块应如何被加载，他的值为"__main__"时表示当前模块是被直接执行。
由于主程序代码无论模块是被导入还是直接被执行都会运行，所以我们需要一种方式在运行时
检测该模块是被导入还是被直接执行。该方式也就是__name__系统变量。如果模块是被导入，
__name__的值为模块名字；如果是被直接执行，__name__的值为"__main__"。
'''
if __name__ == "__main__":
	main()



