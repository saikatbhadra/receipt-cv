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
import time


RECEIPT_PATH = './images/'
WAIT_TIME = 50
DISPLAY = True

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
	
def rect_measure(corners):
	# finds similarity of quad to a rectangle, the lower
	# the number the more it is like a rectangle
	ang1 = ang_lines([corners[0],corners[1]],[corners[2],corners[3]])
	ang2 = ang_lines([corners[0],corners[2]],[corners[1],corners[3]])
	difference = abs(ang1)+abs(ang2)
	
	return difference
	
def ang_lines(line1,line2):
	# calculates angle between two lines
	# lines are defined by pair of tuples which are the end points in x,y
	
	dx1 = line1[1][0]-line1[0][0] + 0.
	dy1 = line1[1][1]-line1[0][1] + 0.
	dx2 = line2[1][0]-line2[0][0] + 0.
	dy2 = line2[1][1]-line2[0][1] + 0.
	
	dot = dx1*dx2 + dy1*dy2
	l = sqrt((dx1**2 + dy1**2) * (dx2**2 + dy2**2))

	return acos(dot/l)

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
	
	# blur to clean up image blur
	image = cv2.medianBlur(image,5)
	image = cv2.GaussianBlur(image,(0,0),3)
	
	image2 = np.array(image,dtype='float32') / 255.
	colorscale = cv2.cvtColor(image2,cv2.COLOR_BGR2HLS)
	
	lightness = colorscale[:,:,1]
	white_bin = np.zeros(np.shape(lightness),dtype='uint8') 
	lightness_win = np.array(lightness*255,dtype='uint8')
	
	print('lightness percentile')
	print(np.percentile(lightness,50))
		
	white_bin[lightness > 0.35] = 255  
	
	if DISPLAY:
		cv2.imshow(window,lightness_win)
		cv2.cv.ResizeWindow(window,960,640)
		cv2.waitKey(WAIT_TIME)
	

	
def area_seg(input_img,window,area_thresh=250,perc_thresh=-1,rank_thresh=-1):
	#TODO change this so area_thresh is relative to window size
	
	# remove salt and pepper noise and smooth minimally
	"""
	strelem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
	blurred = cv2.dilate(image, strelem)
	text = cv2.subtract(blurred,image)  
	"""
	
	norm_dist = sqrt(324*243.)
	input_dist = sqrt(input_img.shape[0]*input_img.shape[1]*1.)
	gauss_blur = int(round(input_dist/norm_dist*3.))

	norm_dist = sqrt(2592*1944.)
	input_dist = sqrt(input_img.shape[0]*input_img.shape[1]*1.)
	median_blur = input_dist/norm_dist*31.
	median_blur = int(round((median_blur-1.)/2)*2+1)
	if median_blur < 3:
		median_blur = -1

	
	image = input_img.copy()
	if median_blur > 1:
		image = cv2.medianBlur(image,median_blur)
	image = cv2.GaussianBlur(image,(0,0),gauss_blur)
	
	if DISPLAY:
		cv2.imshow(window,image)
		cv2.cv.ResizeWindow(window,960,640)
		cv2.waitKey(WAIT_TIME)
	
	grayscale = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	binary = cv2.Canny(grayscale, 10, 20)
	binary = cv2.dilate(binary,None)
	
	
	if DISPLAY:
		cv2.imshow(window,binary)
		cv2.cv.ResizeWindow(window,960,640)
		cv2.waitKey(WAIT_TIME)
	
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
	
	if DISPLAY:
		cv2.imshow(window,binary2)
		cv2.cv.ResizeWindow(window,960,640)
		cv2.waitKey(WAIT_TIME)
	
	return  binary2

def hough_corners(image,window):
	
	debug = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
	lines = cv2.HoughLines(image,1, np.pi/90, 500, None)

	if lines is not None:
		print('Number of lines found')
		print(len(lines[0]))
	linetouse = []
	intersect_pt = []
				
	#TODO: lots of wasted iterations, only do half of combinations
	#TODO: check if lines are almost equivalent (not just exactly equivalent)
	if (lines is not None) and (len(lines) < 200):
		for line1 in lines[0]:
			pts = get_points(line1)
			cv2.line(debug,pts[0],pts[1],(0,255,0),3)
			for line2 in lines[0]:
				if not np.all(line1==line2):
					fit, ang = fit_line(line1,line2,pi*30/180)
					if fit:
						pt = line_intersect(line1,line2)
						if (pt[0] is not None):# and in_image(pt,debug2):
							addtoset(linetouse,line1)
							addtoset(linetouse,line2)
							addtoset(intersect_pt,pt)   
			
	if linetouse is not None:
		for line in linetouse:
			pts = get_points(line)
			cv2.line(debug,pts[0],pts[1],(0,255,0),3)
	
	if intersect_pt is not None:
		for pt in intersect_pt:
			cv2.circle(debug,pt,6,(0,255,0),-1)						 
	
	if linetouse is not None:
		if DISPLAY:
			cv2.imshow(window,debug)
			cv2.cv.ResizeWindow(window,960,640)
			cv2.waitKey()
		
def detect_corners(image,window,area_thresh=250,perc_thresh=-1,rank_thresh=-1):
	
	#downscale image by 8x to focus on key part of image
	norm_area = image.shape[0]*image.shape[1] / (1944. * 2592)
	norm_area = norm_area * 4.
	scale_factor = log(norm_area,2)
	scale_factor = int(round(scale_factor))
	
	if scale_factor < 1:
		scale_factor = 0
	
	down_img = image
	for counter in range(scale_factor):
		down_img = cv2.pyrDown(down_img)
		
	if DISPLAY:
		cv2.namedWindow('output',cv2.cv.CV_WINDOW_NORMAL)
		cv2.imshow('output',down_img)
		cv2.cv.ResizeWindow('output',960,640)
		cv2.waitKey(WAIT_TIME)
	
	# TODO:  adaptive thresholding   
	binary = area_seg(down_img,'output',area_thresh,perc_thresh,rank_thresh)
	
	for counter in range(scale_factor):
		binary = cv2.pyrUp(binary)
		
	# perform hough transform on images
	#ahough_corners(binary,window)
	
	# do contour search	 
	[contour_list,hierarchy] = cv2.findContours(binary.copy(), cv2.RETR_LIST,
	cv2.CHAIN_APPROX_SIMPLE)
	contour_candidate = []
	min_size = int(round(0.2 * binary.shape[0] * binary.shape[1]))
	
	#TODO: change code so instead of taking the largest contour there is a min 
	# and a max 
	edges_color = cv2.cvtColor(binary,cv2.COLOR_GRAY2BGR)
	good_contour = []

	for contour in contour_list:
		# fix the line below to be relative
		if cv2.contourArea(contour)> 100:
			hull = cv2.convexHull(contour)  # find the convex hull of contour
			app_hull = cv2.approxPolyDP(hull,0.05*cv2.arcLength(hull,True),True)
			if len(app_hull)==4:
				area = cv2.contourArea(app_hull)
				if area > min_size:
					good_contour.append(app_hull)
					cv2.drawContours(edges_color,[app_hull],0,(0,255,0),3)
					if DISPLAY:
						cv2.imshow(window,edges_color)
						cv2.waitKey(1)
				else:
					cv2.drawContours(edges_color,[app_hull],0,(0,0,255),3)
					if DISPLAY:
						cv2.imshow(window,edges_color)
						cv2.waitKey(1)
			else:
				cv2.drawContours(edges_color,[app_hull],0,(0,0,255),3)
				if DISPLAY:
					cv2.imshow(window,edges_color)
					cv2.waitKey(1)
	
	contour_candidate = []
	rect_meas = []
	min_rect = pi/2.
	
	final_contour = []
	
	for contour in good_contour:
		corners = fix_corners(contour)
		#TODO: check if convex before this chec,
		if is_convex(corners):
			corners = order_rect(corners)
			contour_candidate.append(corners)
			meas = rect_measure(corners)
			rect_meas.append(meas)
			if meas < min_rect:
				min_rect = meas
				final_contour = corners
	
	print(min_rect)
	
	if DISPLAY:
		cv2.imshow(window,edges_color)
		cv2.cv.ResizeWindow(window,960,640)
		cv2.waitKey()
	
	if final_contour:
		debug = image.copy()
		for corner in final_contour:
			cv2.circle(debug,corner,9,(0,255,0),-3)		
			
		if DISPLAY:
			cv2.imshow(window,debug)
			cv2.cv.ResizeWindow(window,960,640)
			cv2.waitKey()
			
		return final_contour

	return None
	
def dewarp(image,window,corners):
	
	debug = image.copy()

	# draw red line around edges for debug purposes
	cv2.polylines(debug, np.int32([[corners[0],corners[1], corners[3],corners[2]]]),
				  True, (0,255,0),7)

	#show results
	if DISPLAY:
		cv2.imshow(window,debug)
		cv2.cv.ResizeWindow(window,960,640)
		cv2.waitKey(WAIT_TIME)

	# Assemble a rotated rectangle out of that info
	# Todo: move to cV2
	np_corners = np.array(corners)
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
	if DISPLAY:
		cv2.imshow(window,rotated)  
		cv2.cv.ResizeWindow(window,960,640)
		cv2.waitKey(WAIT_TIME)
   
	return rotated

def pre_ocr(image,window,file_name):	
	# code to clean up image prior to OCR step
	
	#convert to grayscale
	grayscale = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	if DISPLAY:
		cv2.imshow(window,grayscale)
		cv2.cv.ResizeWindow(window,960,640)
		cv2.waitKey(WAIT_TIME)
   
	#calculate mask size
	mask_size = min(image.shape[0], image.shape[1])
	mask_size = int(round(mask_size * 0.01))
	mask_size = max(mask_size,3)
	if mask_size % 2 == -0:
		mask_size = mask_size + 1

	binary = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
								   cv2.THRESH_BINARY, mask_size, 5)
								   
	if DISPLAY:
		cv2.imshow(window,binary)
		cv2.cv.ResizeWindow(window,960,640)
		cv2.waitKey()
   
	
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
	
	if DISPLAY:
		cv2.imshow(window,im2)
		cv2.cv.ResizeWindow(window,960,640)
		cv2.waitKey()
		
	
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
	
def is_convex(vertices):
	#checks if within in a list of points that are vertices of a polygon
	# whether the polygon is convex
	
	num_pts = len(vertices)
	z = np.zeros(num_pts)
	
	for pt_idx in range(num_pts):
		pt1 = vertices[pt_idx % (num_pts-1)]
		pt2 = vertices[(pt_idx+1) % (num_pts-1)]
		pt3 = vertices[(pt_idx+2) % (num_pts-1)]
		
		dx1 = pt2[0] - pt1[0]
		dx2 = pt3[0] - pt2[0]
		dy1 = pt2[1] - pt1[1]
		dy2 = pt3[1] - pt2[1]
		z[pt_idx] = dx1*dy2 - dy1*dx2
	
	if z[0] > 0:
		if np.all(z > 0):
			return True
		else:
			return False
	else:
		if np.all(z < 0):
			return True
		else:
			return False
	
def cmp_pt(a,b,center):
	# function to compare if a is to the right of b with relation to point c
	# negative value for less-than,  zero if they are equal, or return a positive value for greater-than
	
	a = (a[0]*1.,a[1]*1.)
	b = (b[0]*1.,b[1]*1.)
	center = (center[0]*1.,center[1]*1.)
	
	if (a[0] >= 0) and (b[0] < 0):
		return -1
	if (a[0] == 0) and (b[0] == 0):
		if a[1] > b[1]:
			return -1
		else:
			return 1
   
	# cross product of center -> a and center -> b
	cross = (a[0]-center[0]) * (b[1]-center[1]) - (b[0]- center[0]) * (a[1] - center[1])
	if (cross < 0):
		return -1
	if (cross > 0):
		return 1

	# points a and b are on the same line from the center
	# check which point is closer to the center
	d1 = (a[0]-center[0]) * (a[0]-center[0]) + (a[1]-center[1]) * (a[1]-center[1])
	d2 = (b[0]-center[0]) * (b[0]-center[0]) + (b[1]-center[1]) * (b[1]-center[1])
	if d1 > d2:
		return -1
	else:
		return 1
	
			
def order_rect(corners):
	# puts order of quad corners such that
	# out = [top_left, top_right, bottom_left, bottom_right]
	# function fails if quad is not convex
	
	if not is_convex(corners):
		print "Quad isn't convex"
		sys.exit(1)
	
	center_x = 0.25 * (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0])
	center_y = 0.25 * (corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1])
	center = (center_x,center_y)
	
	def center_cmp (a,b):
		return cmp_pt(a,b,center)
	
	sorted_corners = sorted(corners,cmp=center_cmp)
	out = [sorted_corners[1],sorted_corners[0],sorted_corners[2],sorted_corners[3]]
	return out

def fix_corners(corners):
	# converts list of lists to list of tuples (why?)   
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
	
	start = time.time()
	
	image = cv2.imread(RECEIPT_PATH + file_name)  
		
	if image==None:
		print "Error opening image"
		sys.exit(1)
		
	orient = get_orient(RECEIPT_PATH + file_name)
	image = fix_orient(image,orient)
	
	#TODO: add border around image to deal with slightly cut off receipts
	#border = [1,1,1,1]
	#border = [x*50 for x in border]
	#image = cv2.copyMakeBorder(image,border[0],border[1],border[2],border[3],cv2.BORDER_CONSTANT)
	
	corners = detect_corners(image,'output',area_thresh=750)
	#if corners is None:
		#corners = detect_corners(image,'output',perc_thresh=95)
		#if corners is None:
			#corners = detect_corners(image,'output',rank_thresh=20)

	if corners:
		rotated = dewarp(image,'output',corners)
		#clean_image = pre_(rotated,'output',file_name)
	else:
		#if no detected corners than remove the border that was inserted before
		pass
		#clean_image = pre_process(image,'output',file_name)
		
	pre_ocr(rotated,'output',file_name)
	
	#ocr_receipt('output',clean_image,file_name)
	# todo check which image is most square like
	
	
	cv2.waitKey()
	cv2.destroyAllWindows()

	elapsed = (time.time() - start)
	print('Time elapsed: ' + str(elapsed))
	return 0	
	
image = scan_receipt('bullitt.jpg')


