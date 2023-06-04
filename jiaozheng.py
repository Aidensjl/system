"""
根据OpenCV-Python Tutorial修改的相机标定程序
OpenCV-Python Tutorial Camera Calibration link：
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration
中文教程链接：
http://woshicver.com/Eighth/7_1_%E7%9B%B8%E6%9C%BA%E6%A0%A1%E5%87%86/
"""

import cv2
import numpy as np
import glob

# 棋盘规格：宽高方向角点个数
w_corner = 9
h_corner = 6

# 角点精准化迭代过程的终止条件
# 迭代次数  
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 世界坐标，如 (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((w_corner*h_corner,3), np.float32) # 矩阵的形式表示棋盘格图像
objp[:,:2] = np.mgrid[0:w_corner,0:h_corner].T.reshape(-1,2)

# 用于存储所有图像的世界坐标点和图像坐标点
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# images = glob.glob('data/images/*.jpg')
images = glob.glob('./pattern.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 寻找棋盘角点
    ret, corners = cv2.findChessboardCorners(gray, (11,11),None)

    # 如果找到，则添加世界坐标点、细化后的图像坐标点
    if ret == True:
        objpoints.append(objp)
        # 亚像素级角点检测（细化）
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # 进行可视化，绘制并显示角点
        img = cv2.drawChessboardCorners(img, (w_corner,h_corner), corners2,ret)
        # 将图像缩小1/4，以便于显示
        img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('img',img)
        cv2.waitKey(500)
cv2.destroyAllWindows()

'''
经过上面的步骤得到了目标点和图像点，下面进行校准。
使用函数 cv.calibrateCamera() 返回相机内参矩阵，畸变系数，旋转矩阵和平移矢量等
其中，mtx：相机内参；dist：畸变系数；revcs：旋转矩阵；tvecs：平移矢量。
'''
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print("相机内参mtx：\n",mtx)
print("畸变系数dist：\n",dist)
print("旋转矩阵rvecs：\n",rvecs)
print("平移矢量tvecs：\n",tvecs)

'''
显示畸变矫正效果
畸变矫正有两种方法：cv.undistort()、remapping
'''
# 读取要畸变矫正的图像
img = cv2.imread("data/images/1.jpg")
h,w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# 使用 cv.undistort() 方法
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# 剪裁图像
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult1.jpg', dst)

# 使用remapping方法
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
# 裁剪图像
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult2.jpg', dst)

# 打印我们要求的两个矩阵参数
# print("newcameramtx:\n",newcameramtx)
# print("dist:\n",dist)

# 重投影误差
tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

print("total error: ", tot_error/len(objpoints))
