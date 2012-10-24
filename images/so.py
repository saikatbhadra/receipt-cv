import cv2
import numpy as np

img = cv2.imread('taco.jpg')
img = cv2.resize(img,(1944,2592))
img = cv2.medianBlur(img,31)
img = cv2.GaussianBlur(img,(0,0),3)

grayscale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
thresh = cv2.Canny(grayscale, 10, 20)
thresh = cv2.dilate(thresh,None)
	
contours,hier = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt)>250:  # remove small areas like noise etc
        hull = cv2.convexHull(cnt)    # find the convex hull of contour
        hull = cv2.approxPolyDP(hull,0.1*cv2.arcLength(hull,True),True)
        if len(hull)==4:
            cv2.drawContours(img,[hull],0,(0,255,0),2)

cv2.namedWindow('output',cv2.cv.CV_WINDOW_NORMAL)
cv2.imshow('output',img)
cv2.cv.ResizeWindow('output',960,640)
cv2.waitKey()
cv2.destroyAllWindows()
