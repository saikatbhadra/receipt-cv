#!/usr/bin/python
import cv
import scipy
from scipy import ndimage
import tesseract
import matplotlib.pyplot as plt

RECEIPT_PATH = "/media/psf/Home/Dropbox/Receipts/"


def scanReceipt(file_name):

    #load image - eventually pass as argument to python
    image = cv.LoadImage(RECEIPT_PATH + file_name)
    image_size = cv.GetSize(image)

    if not image:
        print "Error opening image"
        sys.exit(1)

    #cv.NamedWindow('output',cv.CV_WINDOW_NORMAL)
    #cv.ShowImage('output',image)
    #cv.WaitKey()

    #convert to grayscale
    grayscale = cv.CreateImage(cv.GetSize(image),8,1)
    cv.CvtColor(image,grayscale,cv.CV_RGB2GRAY)

    #cv.ShowImage('output',grayscale)
    #cv.WaitKey()

    #binarize
    mask_size = min(image_size[0], image_size[1])
    mask_size = int(round(mask_size * 0.01))
    mask_size = max(mask_size,3)
    if mask_size % 2 == -0:
        mask_size = mask_size + 1
    
    
    binary = cv.CreateImage(cv.GetSize(grayscale),cv.IPL_DEPTH_8U,1)
    cv.AdaptiveThreshold(grayscale, binary, 255,
                         cv.CV_ADAPTIVE_THRESH_GAUSSIAN_C,
                         cv.CV_THRESH_BINARY, mask_size, 10)
    #cv.ShowImage('output',binary)
    #cv.WaitKey()

    
    #pre-processing [remove blobs of size 1]
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
    struct2 = ndimage.generate_binary_structure(2,2)
    im2 = ndimage.binary_dilation(im,structure=struct2,iterations=1).astype(im.dtype)

    #clean up large blobs
    blob2, num_blob2 = ndimage.label(im2,structure=mask)
    blob2_sizes = ndimage.sum(im, blob2, range(num_blob2 + 1))

    large_blob = blob2_sizes > 200e3
    remove_large_blob = large_blob[blob2]
    #im2[remove_large_blob] = 0

  
    #cv.DestroyWindow('output')

    #write to image
    scipy.misc.imsave('FinalBin' + file_name,im2)
    image = cv.LoadImage('FinalBin' + file_name)
   

    #do OCR
    api = tesseract.TessBaseAPI()
    api.Init(".", "eng", tesseract.OEM_DEFAULT)
    api.SetPageSegMode(tesseract.PSM_AUTO)
    tesseract.SetCvImage(image,api)
    text=api.GetUTF8Text()
    conf=api.AllWordConfidences()

    print(type(text))
    print 'Output of OCR machine:'
    print text
   
    #cv.WaitKey()
    #cv.SaveImage('TempBin' + file_name, binary)
    #txt =

    #be sure to close window
    
    
 
scanReceipt('taco.jpg')



#if __name__ == "__main__":


    
