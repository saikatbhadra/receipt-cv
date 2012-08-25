#!/usr/bin/python
import cv
import cv2
import numpy as np
import scipy
from scipy import ndimage
import tesseract
import os
import subprocess

RECEIPT_PATH = "/media/psf/Home/Dropbox/Receipts/"
   
def mouse_call(event,x,y,flags,param):
    if (event == cv.CV_EVENT_LBUTTONUP) and (len(param) < 4):
        print 'Click Detected!'
        param.append((x,y))

def detect_corners(image):
    
    cv2.namedWindow('img',cv2.cv.CV_WINDOW_NORMAL)
    cv2.cv.ResizeWindow('img',960,640)

    print(type(image))
    image_size = [image.shape[0], image.shape[1]]
    clicked_corners = [(408,69), (1912,291), (72,2186),(1584, 2426)]

    grayscale = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #grayscale = cv2.equalizeHist(grayscale)

    cv2.imshow('img',grayscale)
    cv2.waitKey()
  
    mask_size = min(image_size[0], image_size[1])
    mask_size = int(round(mask_size * 0.01))
    mask_size = max(mask_size,3)
    print mask_size
    if mask_size % 2 == -0:
        mask_size = mask_size + 1
       
    #binary  = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
         #                 cv2.THRESH_BINARY, mask_size, 5)

    ret,binary = cv2.threshold(grayscale,127,255,0)

    #binary = cv2.Canny(grayscale,100,150)

    cv2.imshow('img',binary)
    cv2.waitKey()

    contours,hier = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt)>5:  # remove small areas like noise etc
            hull = cv2.convexHull(cnt)    # find the convex hull of contour
            hull = cv2.approxPolyDP(hull,0.1*cv2.arcLength(hull,True),True)
            if len(hull)>=3:
                cv2.drawContours(image,[hull],0,(0,255,0),2)

    cv2.imshow('img',image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    """clicked_corners = []
     cv.SetMouseCallback(window,mouse_call,clicked_corners)
    cv.WaitKey()"""
    return clicked_corners
    
def dewarp(image,window,clicked_corners):
    debug = cv.CloneImage(image)

    # draw red line around edges for debug purposes
    cv.PolyLine(debug, [[clicked_corners[0],clicked_corners[1],
                         clicked_corners[3],clicked_corners[2]]],
                True, cv.RGB(0,255,0),7)
    
    cv.ShowImage(window,debug)
    cv.WaitKey()
    
    # Assemble a rotated rectangle out of that info
    #rot_box = cv.MinAreaRect2(corners)
    enc_box = cv.BoundingRect(clicked_corners)
    new_corners = [(0,0),(enc_box[2]-1,0),(0,enc_box[3]-1),(enc_box[2]-1,enc_box[3]-1)]  

    warp_mat = cv.CreateMat(3,3,cv.CV_32FC1)
    cv.GetPerspectiveTransform(clicked_corners,new_corners,warp_mat)

    rotated = cv.CloneImage(image)
    cv.WarpPerspective(image,rotated,warp_mat)
          
    cv.ShowImage(window,rotated)
    cv.WaitKey(10)
   
    return rotated

def pre_process(image,window,file_name):

    image_size = cv.GetSize(image)    
    
    #convert to grayscale
    grayscale = cv.CreateImage(cv.GetSize(image),8,1)
    cv.CvtColor(image,grayscale,cv.CV_RGB2GRAY)

    #equalize histogram
    cv.EqualizeHist(grayscale,grayscale)

    cv.ShowImage(window,grayscale)
    cv.WaitKey()
    
    #calculate mask size
    mask_size = min(image_size[0], image_size[1])
    mask_size = int(round(mask_size * 0.01))
    mask_size = max(mask_size,3)
    if mask_size % 2 == -0:
        mask_size = mask_size + 1
       
    binary = cv.CreateImage(cv.GetSize(grayscale),cv.IPL_DEPTH_8U,1)
    cv.AdaptiveThreshold(grayscale, binary, 255,
                         cv.CV_ADAPTIVE_THRESH_GAUSSIAN_C,
                         cv.CV_THRESH_BINARY, mask_size, 10)

    cv.ShowImage(window,binary)
    cv.WaitKey()
    
    #read the image in numpy
    #TODO: fix it so we don't have to write to disk
    cv.SaveImage('TempBin' + file_name, binary)
    im = scipy.misc.imread('TempBin' + file_name,flatten=1)
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
    scipy.misc.imsave('Temp2Bin' + file_name,im)
    scipy.misc.imsave('FinalBin' + file_name,im2)
    final_bin = cv.LoadImage('FinalBin' + file_name)
    temp_bin = cv.LoadImage('Temp2Bin' + file_name)
  
    cv.ShowImage(window,temp_bin)
    cv.WaitKey()
    
    cv.ShowImage(window,final_bin)
    cv.WaitKey()
    
def ocr_receipt(window,file_name):
    string_int=('tesseract ' + 'TempBin' + file_name + ' '
                    + RECEIPT_PATH + file_name[:-4] + '_int1')
    string_int2=('tesseract ' + 'TempBin2' + file_name + ' '
                    + RECEIPT_PATH + file_name[:-4] + '_int2')
    string_final = ('tesseract ' + 'Final.Bin' + file_name + ' '
                    + RECEIPT_PATH + file_name[:-4] + '_final')
    
    temp = subprocess.call(string_int,shell=True)
    temp = subprocess.call(string_int2,shell=True)
    temp = subprocess.call(string_final,shell=True)
    print string_final
    
    """ api = tesseract.TessBaseAPI()
    api.Init(".", "eng", tesseract.OEM_DEFAULT)
    api.SetPageSegMode(tesseract.PSM_AUTO)
    tesseract.SetCvImage(image,api)
    text=api.GetUTF8Text()
    conf=api.AllWordConfidences()

    text_file = open(RECEIPT_PATH + file_name[:-4] + '.txt','w')
    text_file.write(text)
    text_file.close() """
           
       
def scan_receipt(file_name):
    image = cv.LoadImage(RECEIPT_PATH + file_name)
    if not image:
        print "Error opening image"
        sys.exit(1)

    cv.NamedWindow('output',cv.CV_WINDOW_NORMAL)
    cv.ShowImage('output',image)
    cv.ResizeWindow('output',960,640)
    cv.WaitKey()
    image2 = cv2.imread(RECEIPT_PATH + file_name)
    corners = detect_corners(image2)
    
    #rotated = dewarp(image,'output',corners)
    #pre_process(rotated,'output',file_name)
    #ocr_receipt('output',file_name)
    
     
    
scan_receipt('super.jpg')






    
