#!/usr/bin/python
import cv2
import cv
import numpy as np
from scipy import ndimage
import tesseract
import scipy
from math import *

RECEIPT_PATH = "/media/psf/Home/Dropbox/Receipts/"
WAIT_TIME = 500

def mouse_call(event,x,y,flags,param):
    if (event == cv.CV_EVENT_LBUTTONUP) and (len(param) < 4):
        print 'Click Detected!'
        param.append((x,y))

def detect_corners(image,window):
    
    debug = image.copy()
    image_size = [image.shape[0], image.shape[1]]
   
    #blur image to clean up noise prior to canny
    #blurred = cv2.medianBlur(image,7)
    blurred = cv2.GaussianBlur(image,(0,0),3)

    cv2.imshow(window,blurred)
    cv2.cv.ResizeWindow(window,960,640)
    cv2.waitKey(WAIT_TIME)
    
    #TODO: eventually loop through channels
    grayscale = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
    cv2.imshow(window,grayscale)
    cv2.cv.ResizeWindow(window,960,640)
    cv2.waitKey(WAIT_TIME)

    # try canny method and then adaptive thresholding    
    # TODO:  adaptive thresholding     
    binary = cv2.Canny(grayscale, 10, 20)
    binary = cv2.dilate(binary,None)

    #elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    #dil = cv2.dilate(grayscale,None)
    #dil = cv2.dilate(grayscale,None) 
    #ero = cv2.erode(grayscale,None)
    #retval,binary = cv2.threshold(dil-ero,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #mask_size = min(grayscale.shape[0], grayscale.shape[1])
    #mask_size = int(round(mask_size * 0.01))
    #mask_size = max(mask_size,3)
    #if mask_size % 2 == -0:
    #    mask_size = mask_size + 1
    #binary = cv2.adaptiveThreshold(dil-ero, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                               cv2.THRESH_BINARY, mask_size, 0)
    #binary = cv2.erode(binary,None)
    #binary = cv2.erode(binary,None)
    
    cv2.imshow(window,binary)
    cv2.cv.ResizeWindow(window,960,640)
    cv2.waitKey(WAIT_TIME)

    debug = image.copy()
    lines = cv2.HoughLines(binary.copy(),1, np.pi/180, 1000)
    print lines
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(round(x0 + 1000*(-b)))
        y1 = int(round(y0 + 1000*(a))) 
        x2 = int(round(x0 - 1000*(-b)))  
        y2 = int(round(y0 - 1000*(a)))
        cv2.line(debug,(x1,y1),(x2,y2),(0,255,0),3)
    
    #show results
    cv2.imshow(window,debug)
    cv2.cv.ResizeWindow(window,960,640)
    cv2.waitKey(WAIT_TIME)
    
    [contour_list,hierarchy] = cv2.findContours(binary, cv2.RETR_LIST,
                                                cv2.CHAIN_APPROX_SIMPLE)
    contour_candidate = []
    max_size = 0
    min_size = 500**2

    for contour in contour_list:
        if cv2.contourArea(contour)> 500:
            hull = cv2.convexHull(contour)    # find the convex hull of contour
            hull = cv2.approxPolyDP(hull,0.1*cv2.arcLength(hull,True),True)
            if len(hull)==4:
                cv2.drawContours(debug,[hull],0,(0,0,255),3)
                if ((cv2.contourArea(hull) > max_size)
                    and (cv2.contourArea(hull) > min_size)):
                    max_size = cv2.contourArea(hull)
                    contour_candidate = [contour, hull]

    cv2.imshow(window,debug)
    cv2.cv.ResizeWindow(window,960,640)
    cv2.waitKey(WAIT_TIME)

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
    cv2.waitKey(WAIT_TIME)
   
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
    cv2.waitKey()
    
    #read the image in numpy
    #TODO: fix it so we don't have to write to disk
    cv2.imwrite('TempBin' + file_name, binary)
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
    final_bin = cv2.imread('FinalBin' + file_name)
    temp_bin = cv2.imread('Temp2Bin' + file_name)

    cv2.imshow(window,temp_bin)
    cv2.cv.ResizeWindow(window,960,640)
    cv2.waitKey(WAIT_TIME)

    cv2.imshow(window,final_bin)
    cv2.cv.ResizeWindow(window,960,640)
    cv2.waitKey(WAIT_TIME)

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
    
       
def scan_receipt(file_name):

    image = cv2.imread(RECEIPT_PATH + file_name)
    
    if image==None:
        print "Error opening image"
        sys.exit(1)

    image = cv2.resize(image,(1944,2592))

    cv2.namedWindow('output',cv2.cv.CV_WINDOW_NORMAL)
    cv2.imshow('output',image)
    cv2.cv.ResizeWindow('output',960,640)
    cv2.waitKey(WAIT_TIME)
    
    corners = detect_corners(image,'output')
    if corners:
        rotated = dewarp(image,'output',corners)
        clean_image = pre_process(rotated,'output',file_name)
    else:
        clean_image = pre_process(image,'output',file_name)
    #ocr_receipt('output',clean_image,file_name)

    cv2.destroyAllWindows()

    print 'Done!'
    return clean_image
    
    
image = scan_receipt('taco.jpg')






    
