#!/usr/bin/python
import cv2
import cv
import numpy as np
from scipy import ndimage
import tesseract
import scipy
from math import *
import os
import sys
import pyexiv2
import matplotlib.pyplot as plt

RECEIPT_PATH = './images/'
WAIT_TIME = 500

def addtoset(self,addelem):
	for elem in self:
		if np.all(elem == addelem):
			return
	self.append(addelem)

def get_points(line):
	rho = line[0]
	theta = line[1]
	alpha = 10000
	
	a = cos(theta)
	b = sin(theta)	
	x0 = a*rho
	y0 = b*rho
	
	x1 = int(round(x0 - alpha*(b)))
	y1 = int(round(y0 + alpha*(a)))
	x2 = int(round(x0 + alpha*(b)))  
	y2 = int(round(y0 - alpha*(a)))
	
	return [(x1,y1),(x2,y2)]

def fit_line(line1, line2, min_theta):
	
	theta1 = line1[1]
	theta2 = line2[1]
	
	if theta1 < min_theta:
		theta1 += np.pi
	
	if theta2 < min_theta:
		theta2 += np.pi
	
	fit = abs(theta1 - theta2) > min_theta
	ang = abs(theta1 - theta2)
	
	return fit, ang

def in_image(pt,image):
	if pt[0] >= image.shape[0]:
		return False
	if pt[1] >= image.shape[1]:
		return False
	return True
	
def line_intersect(line1, line2):
	
	rho1 = line1[0]
	rho2 = line2[0]
	theta1 = line1[1]
	theta2 = line2[1]
	
	a = np.array([[cos(theta1), sin(theta1)],[cos(theta2), sin(theta2)]])
	b = np.array([rho1,rho2])	
	
	det = np.linalg.det(a)
	
	if abs(det) > 0:
		pts = np.linalg.solve(a,b)
		pts = np.int16(np.round(pts))
		return (pts[0],pts[1])
	else:
		return (None,None)
		
def color_seg(image,window):
	# median blur will remove out some text
	blurred = cv2.medianBlur(image.copy(),35)	
	#blurred = cv2.pyrMeanShiftFiltering(blurred,20,100,None,4)
	
def text_seg(input_img,window,area_thresh=250,perc_thresh=-1,rank_thresh=-1):
	
	# remove salt and pepper noise and smooth minimally
	"""
	image = cv2.medianBlur(input_img,3)
	image = cv2.GaussianBlur(image,(0,0),3)
	
	cv2.imshow(window,image)
	cv2.cv.ResizeWindow(window,960,640)
	cv2.waitKey(WAIT_TIME)
	
	strelem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
	blurred = cv2.dilate(image, strelem)
	text = cv2.subtract(blurred,image)	
	
	cv2.imshow(window,text)
	cv2.cv.ResizeWindow(window,960,640)
	cv2.waitKey(WAIT_TIME)
	
	text_gray = cv2.cvtColor(text,cv2.COLOR_BGR2GRAY)
	
	cv2.imshow(window,text_gray)
	cv2.cv.ResizeWindow(window,960,640)
	cv2.waitKey(WAIT_TIME)
	
	thres= int(round(np.percentile(text_gray,99)))
	
	retval,binary = cv2.threshold(text_gray,thres,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
			
	cv2.imshow(window,binary)
	cv2.cv.ResizeWindow(window,960,640)
	cv2.waitKey() """
	
	image = cv2.medianBlur(input_img,5)
	image = cv2.GaussianBlur(image,(0,0),3)
	cv2.imshow(window,image)
	cv2.cv.ResizeWindow(window,960,640)
	cv2.waitKey(WAIT_TIME)
	
	image2 = np.array(image,dtype='float32') / 255.
	colorscale = cv2.cvtColor(image2,cv2.COLOR_BGR2HLS)
	
	lightness = colorscale[:,:,1]
	white_bin = np.zeros(np.shape(lightness),dtype='uint8')	
	lightness_win = np.array(lightness*255,dtype='uint8')
	
	print('lightness')
	print(np.percentile(lightness,50))
	
	print('now showing!')
	
	white_bin[lightness > 0.35] = 255  
	cv2.imshow(window,lightness_win)
	cv2.cv.ResizeWindow(window,960,640)
	cv2.waitKey()
	
	
	grayscale = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	binary = cv2.Canny(grayscale, 10, 20)
	binary = cv2.dilate(binary,None)
	
	cv2.imshow(window,binary)
	cv2.cv.ResizeWindow(window,960,640)
	cv2.waitKey()
	
	# remove small blobs	
	mask = [[1,1,1],[1,1,1],[1,1,1]]
	blobs, num_blobs = ndimage.label(binary,structure=mask)
	areas = ndimage.sum(binary,blobs,range(num_blobs+1))	
	areas = areas / 255
	binary2 = binary.copy()

	if rank_thresh > 0:
		perc_thresh = (1 - rank_thresh / num_blobs)*100
		area_thresh = np.percentile(areas,perc_thresh)
		area_thresh = max(area_thresh,250)
		small_bin = areas < area_thresh
		pass
	elif perc_thresh > 0:
		area_thresh = np.percentile(areas,perc_thresh)
		area_thresh = max(area_thresh,250)
		small_bin = areas < area_thresh
	else:
		area_thresh = max(area_thresh,250)
		small_bin = areas < area_thresh

	remove_blobs = small_bin[blobs]
	binary2[remove_blobs] = 0	
	
	cv2.imshow(window,binary2)
	cv2.cv.ResizeWindow(window,960,640)
	cv2.waitKey()
	
	#calculate bounding box between
	return  binary2
		
def detect_corners(image,window,area_thresh=250,perc_thresh=-1,rank_thresh=-1):
	
	"""debug = image.copy()
	image_size = [image.shape[0], image.shape[1]]
	
	# attempts to remove some txt (salt and pepper noise)using median
	blurred = cv2.medianBlur(image.copy(),11)
	
	#blur image to clean up noise prior to canny
	blurred = cv2.GaussianBlur(blurred,(0,0),3)
	cv2.imshow(window,blurred)
	cv2.cv.ResizeWindow(window,960,640)
	cv2.waitKey()
	
	#TODO: grayscale conversion
	grayscale = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
	cv2.imshow(window,grayscale)
	cv2.cv.ResizeWindow(window,960,640)
	cv2.waitKey()
	
	# try canny method and then adaptive thresholding	
	# TODO:  adaptive thresholding	 
	binary = cv2.Canny(grayscale, 10, 20)
	binary = cv2.dilate(binary,None)
	binary = cv2.dilate(binary,None)
	#binary = cv2.dilate(binary,None)

	cv2.imshow(window,binary)
	cv2.cv.ResizeWindow(window,960,640)
	cv2.waitKey()"""
	
	binary = text_seg(image,'output',area_thresh,perc_thresh,rank_thresh)
			
	# perform hough transform on images
	debug = image.copy() 
	
	lines = cv2.HoughLines(binary,1, np.pi/90, 1000, None)
	if lines is not None:
		print(len(lines[0]))
	linetouse = []
	intersect_pt = []
	
	debug2 = debug.copy()
	
	#TODO: lots of wasted iterations, only do half of combinations
	#TODO: check if lines are almost equivalent (not just exactly equivalent)
	if lines is not None:
		for line1 in lines[0]:
			pts = get_points(line1)
			cv2.line(debug2,pts[0],pts[1],(0,255,0),3)
			for line2 in lines[0]:
				if not np.all(line1==line2):
					fit, ang = fit_line(line1,line2,pi*30/180)
					if fit:
						pt = line_intersect(line1,line2)
						if (pt[0] is not None):# and in_image(pt,debug2):
							addtoset(linetouse,line1)
							addtoset(linetouse,line2)
							addtoset(intersect_pt,pt)	
	
	cv2.imshow(window,debug2)
	cv2.cv.ResizeWindow(window,960,640)
	cv2.waitKey()
	
	if linetouse is not None:
		for line in linetouse:
			pts = get_points(line)
			cv2.line(debug,pts[0],pts[1],(0,0,255),3)
	
	if intersect_pt is not None:
		for pt in intersect_pt:
			cv2.circle(debug,pt,6,(0,255,0),-1)							
	
	cv2.imshow(window,debug)
	cv2.cv.ResizeWindow(window,960,640)
	cv2.waitKey()
			
	[contour_list,hierarchy] = cv2.findContours(binary, cv2.RETR_LIST,
												cv2.CHAIN_APPROX_SIMPLE)
	contour_candidate = []
	max_size = 0
	min_size = 500**2

	for contour in contour_list:
		if cv2.contourArea(contour)> 500:
			hull = cv2.convexHull(contour)	# find the convex hull of contour
			hull = cv2.approxPolyDP(hull,0.1*cv2.arcLength(hull,True),True)
			if len(hull)==4:
				cv2.drawContours(debug,[hull],0,(0,0,255),3)
				if ((cv2.contourArea(hull) > max_size)
					and (cv2.contourArea(hull) > min_size)):
					max_size = cv2.contourArea(hull)
					contour_candidate = [contour, hull]

	cv2.imshow(window,debug)
	cv2.cv.ResizeWindow(window,960,640)
	cv2.waitKey()

	if contour_candidate:
		corners = fix_corners(contour_candidate[1])
		corners = order_rect(corners)

		debug2 = image.copy()
		for corner in corners:
			cv2.circle(debug2,corner,9,(0,255,0),-3)

		cv2.imshow(window,debug2)
		cv2.cv.ResizeWindow(window,960,640)
		cv2.waitKey(WAIT_TIME)
		return corners

	return None
	
def dewarp(image,window,corners):
	
	debug = image.copy()

	# draw red line around edges for debug purposes
	cv2.polylines(debug, np.int32([[corners[0],corners[1], corners[3],corners[2]]]),
				  True, (0,255,0),7)

	#show results
	cv2.imshow(window,debug)
	cv2.cv.ResizeWindow(window,960,640)
	cv2.waitKey(WAIT_TIME)

	
	# Assemble a rotated rectangle out of that info
	rot_box = cv.MinAreaRect2(corners)
	enc_box = cv.BoundingRect(corners)

	scaling = 1.0
	border = 10
	pt_x = enc_box[2]*scaling
	pt_y = enc_box[3]*scaling

	new_corners = [(border,border),(pt_x-1+border,border),
				   (border,pt_y-1+border),(pt_x-1+border,pt_y-1+border)]

	corners = np.array(corners,np.float32)
	new_corners = np.array(new_corners,np.float32)
   
	warp_mat = cv2.getPerspectiveTransform(corners, new_corners)
	rotated = cv2.warpPerspective(image, warp_mat, (int(round(pt_x+border*2)),
													int(round(pt_y+border*2))))
 
	#show results
	cv2.imshow(window,rotated)  
	cv2.cv.ResizeWindow(window,960,640)
	cv2.waitKey()
   
	return rotated

def pre_process(image,window,file_name):

	image_size = [image.shape[0], image.shape[1]]
	
	#convert to grayscale
	grayscale = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	cv2.imshow(window,grayscale)
	cv2.cv.ResizeWindow(window,960,640)
	cv2.waitKey(WAIT_TIME)

   
	#calculate mask size
	mask_size = min(image_size[0], image_size[1])
	mask_size = int(round(mask_size * 0.01))
	mask_size = max(mask_size,3)
	if mask_size % 2 == -0:
		mask_size = mask_size + 1

	binary = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
								   cv2.THRESH_BINARY, mask_size, 5)

	cv2.imshow(window,binary)
	cv2.cv.ResizeWindow(window,960,640)
	cv2.waitKey(WAIT_TIME)
	
	#read the image in numpy
	im = np.array(binary)
	im[im>100]=255
	im[im<=100]=0
	im = 255 - im

	mask = [[1,1,1],
		 [1,1,1],
		 [1,1,1]]

	blob, num_blob = ndimage.label(im,structure=mask)
	blob_sizes = ndimage.sum(im, blob, range(num_blob + 1))
	
	#clean up small blobs
	small_blob = blob_sizes < 5e3
	remove_small_blob = small_blob[blob]
	im[remove_small_blob] = 0
	
	#connect large blobs
	large_mask = ndimage.generate_binary_structure(2,2)
	im2 = ndimage.binary_dilation(im,structure=large_mask,iterations=1).astype(im.dtype)

	#write file
	scipy.misc.imsave('FinalBin' + file_name,im2)
	cv2.imshow(window,im2)
	cv2.cv.ResizeWindow(window,960,640)
	cv2.waitKey()

	return final_bin		
	
def ocr_receipt(window,image,file_name):
	"""string_int=('tesseract ' + 'TempBin' + file_name + ' '
					+ RECEIPT_PATH + file_name[:-4] + '_int1')
	string_int2=('tesseract ' + 'TempBin2' + file_name + ' '
					+ RECEIPT_PATH + file_name[:-4] + '_int2')
	string_final = ('tesseract ' + 'Final.Bin' + file_name + ' '
					+ RECEIPT_PATH + file_name[:-4] + '_final')
	
	temp = subprocess.call(string_int,shell=True)
	temp = subprocess.call(string_int2,shell=True)
	temp = subprocess.call(string_final,shell=True)
	os.system(string_final)
	print string_final"""
	
	api = tesseract.TessBaseAPI()
	api.Init(".", "eng", tesseract.OEM_DEFAULT)
	api.SetPageSegMode(tesseract.PSM_AUTO)
	tesseract.SetCvImage(image,api)
	text=api.GetUTF8Text()
	conf=api.AllWordConfidences()

	print text

	text_file = open(RECEIPT_PATH + file_name[:-4] + '.txt','w')
	text_file.write(text)
	text_file.close()

def order_rect(corners):

	distance = []
	
	#calculate distance for each point
	for pt in corners:
		dist = sqrt(pt[0]**2 + pt[1]**2)
		distance.append(dist)

	top_left = corners[distance.index(min(distance))]
	bottom_right = corners[distance.index(max(distance))]

	corners2 = corners
	corners2.remove(top_left)
	corners2.remove(bottom_right)

	xdist = []

	for pts in corners2:
		xdist.append(fabs(top_left[0] - pts[0]))

	top_right = corners2[xdist.index(max(xdist))]
	corners2.remove(top_right)
	bottom_left = corners2[0]

	out = [top_left, top_right, bottom_left, bottom_right]
	return out

def fix_corners(corners):
	out = []
	for corner in corners:
		out.append((corner[0][0],corner[0][1]))
	return out
	
def get_orient(file_name):
	metadata = pyexiv2.ImageMetadata(file_name)
	metadata.read()
	try:
		tag = metadata['Exif.Image.Orientation']
		value = tag.value
	except KeyError:
		value = 1
	return value
	
def fix_orient(image,value):
	
	if value <= 1:
		#do nothing
		out = image.copy()
	elif value == 2:
		#flip image horizontally
		out = cv2.flip(image,1)
	elif value == 3:
		#flip vertically, horizontally or rotate 180
		out = cv2.flip(image,-1)
	elif value == 4:
		#flip vertically
		out = cv2.flip(image,0)
	elif value == 5:
		# transpose
		out = cv2.transpose(image)
	elif value == 6:
		# flip vertically, transpose or rotate 90
		temp = cv2.flip(image,0)
		out = cv2.transpose(temp)
	elif value == 7:
		# flip horizontally, vertically, transpose or transverse
		temp = cv2.flip(image,-1)
		out = cv2.transpose(temp)
	elif value == 8:
		# flip horizontally, transpose or rotate 270
		temp = cv2.flip(image,1)
		out = cv2.transpose(temp)	
	return out
	
	
def scan_receipt(file_name):

	image = cv2.imread(RECEIPT_PATH + file_name)  
	
	if image==None:
		print "Error opening image"
		sys.exit(1)
		
	orient = get_orient(RECEIPT_PATH + file_name)
	image = fix_orient(image,orient)
	image = cv2.resize(image,(1944,2592))
	
	cv2.namedWindow('output',cv2.cv.CV_WINDOW_NORMAL)
	cv2.imshow('output',image)
	cv2.cv.ResizeWindow('output',960,640)
	cv2.waitKey(WAIT_TIME)
	
	
	#color_seg(image,'output')	
	corners = detect_corners(image,'output',area_thresh=250)
	if corners is None:
		corners = detect_corners(image,'output',perc_thresh=98)
		if corners is None:
			corners = detect_corners(image,'output',rank_thresh=20)

	if corners:
		rotated = dewarp(image,'output',corners)
		#clean_image = pre_process(rotated,'output',file_name)
	else:
		pass
		#clean_image = pre_process(image,'output',file_name)
	#ocr_receipt('output',clean_image,file_name)

	cv2.destroyAllWindows()

	print 'Done!'
	return 0	
	
image = scan_receipt('bullitt.JPG')






	
