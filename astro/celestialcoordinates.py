import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.colors import LogNorm
from astropy.visualization import (MinMaxInterval, SqrtStretch, ImageNormalize)
from astropy.visualization import ImageNormalize
import astroalign as aa
from collections import deque
from PIL import Image
import os
from astropy.io import ascii
from scipy.signal import convolve as scipy_convolve
from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel
from astropy.stats import sigma_clipped_stats
import glob
from astropy.stats import SigmaClip
from photutils.background import Background2D
from photutils.segmentation import SegmentationImage
from photutils.background import MedianBackground
from importlib import reload
import cv2
import imutils
from astropy.coordinates import SkyCoord
from astropy import units as u
import subprocess
import shutil
from photutils.utils import circular_footprint
import argparse
import sep
from astropy.wcs import WCS
import sunpy.coordinates
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import wcs_to_celestial_frame
from astropy.wcs import WCS, FITSFixedWarning
import warnings


def histogram1():
    question = input('Has any error occurred? If so, answer YES')
    brightness = input('Brightness is automatically set up. If the image is too bright, input here the number that you would like it to be. Take note, that the greater the value gets, the brighter it will be: If you do not wish to change the brightness, input "NO"')
    if brightness == 'NO':
        b = 45
    else:
        b = int(brightness)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message = "'datfix' made the change",
                                category=FITSFixedWarning)
    if question == 'YES':
        #image_data2 = np.where(image_data > 0.0000000001, image_data, -10)
        img_stretch = image_data * np.log10(image_data + 0.1)#, where=image_data > 0)# casting = 'unsafe')# out=image_data2, where=image_data2 > 0, casting = 'unsafe')
        histogram = plt.hist(np.concatenate(img_stretch), bins=100*np.linspace(0,256), log=True)
        norm = ImageNormalize(img_stretch, interval=MinMaxInterval(), stretch=SqrtStretch())
        fig = plt.figure()
        wcs = WCS(hdu_list[0].header)
        ax = fig.add_subplot(111, projection = wcs)

        if np.min(img_stretch) <= 0:
            vmin = np.min(img_stretch)+1
        elif np.min(img_stretch) > 0:
            vmin = np.min(img_stretch)
        im = ax.imshow(img_stretch, origin='lower',
            norm = LogNorm(vmin, vmax=np.max(img_stretch)*(1/b)),
            cmap='gray')
        fig.colorbar(im)
    else:
        img_stretch = image_data * np.log10(image_data)
        histogram = plt.hist(np.concatenate(img_stretch), bins=100*np.linspace(0,256), log=True)
        norm = ImageNormalize(img_stretch, interval=MinMaxInterval() ,stretch=SqrtStretch())
        fig = plt.figure()
        wcs = WCS(hdu_list[0].header)
        ax = fig.add_subplot(111, projection = wcs)
        im = ax.imshow(img_stretch, origin = 'lower',
            norm=LogNorm(vmin = np.min(img_stretch), vmax = np.max(img_stretch)*(1/b)),
            cmap="gray") #grey/binary
        fig.colorbar(im)
    return im


def blob_finder3(IMAGE_DATA):
    	image_gray = IMAGE_DATA
    # 	image_gray = rgb2gray(image)

    	blobs_doh = blob_doh(
    		image_gray,
    		min_sigma=30,
    		max_sigma=90,
    		num_sigma=5,
    		threshold=40,
    		overlap=0.25)

    	blobs_list = [blobs_doh]
    	colors = ["red"]
    	titles = ["Determinant of Hessian"]
    	sequence = zip(blobs_list, colors, titles)
    	fig, ax = plt.subplots()

    	for idx, (blobs, color, title) in enumerate(sequence):
                if question == 'YES':
                    contingency = image_gray * np.log10(image_gray, where=image_gray > 0, casting = 'unsafe')
                    if np.min(image_gray * np.log10(image_gray)) <= 0:
                        vmin = np.min(contingency)
                    elif np.min(image_gray * np.log10(image_gray)) > 0:
                        vmin = np.min(image_gray * np.log10(image_gray))
                    ax.imshow(contingency, origin = 'lower',
                        norm = LogNorm(vmin, vmax=np.max(contingency)*1/45),
                        cmap = 'gray')
                else:
                    ax.imshow(contingency, origin = 'lower',
                    norm=LogNorm(vmin, vmax = np.max(image_gray * np.log10(image_gray))*(1/45)),
                    cmap="gray")
        		for blob in blobs:
        			y, x, r = blob
        			c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        			ax.add_patch(c)
        		ax.set_axis_off()

# 	plt.tight_layout()
	   plt.show()

def click_event(event, x, y, flags, params):
    #checking for left mouse clicks
    event =
    if event == cv2.EVENT_LBUTTONDOWN:
        #displaying coordinates.
        print(x, '', y)
        #displaying coordinates on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image_data, str(x) + ',' + str(y), (x,y), font, 1, (255, 0, 0), 2)
        cv2.imshow('Astronomical image', image_data)
    if event == cv2.EVENT_RBUTTONDOWN:
        #displaying the coordinates
        print(x, '', y)
        #displaying the coordinates on the image window.
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = image_data[y, x, 0]
        g = image_data[y, x, 1]
        r = image_data[y, x, 2]

        cv2.putText(image_data, srt(b) + ',' + str(g) + ',' + str(r), (x,y), font, 1, (255, 255, 0), 2)
        cv2.imshow('Astronomical image', image_data)

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

source = input('Enter the name of the Science image:')
hdu_list = fits.open(source)
a = hdu_list.info()
b = hdu_list[0].header

#this is the reference pixel and coordinate
while True:
    try:
        print(b['CRVAL1'], b['CRVAL2'])
        print(b['CRPIX1'], b['CRPIX2'])
        break
    except:
        print('Value not found')
    break
#This is the pixel resoltuion (at the reference pixel).
while True:
    try:
        print(header['CDELT1'], header['CDELT2'])
        break
    except:
        print('Value not found')
    break
#This is the rotation angle, in degress (at the reference pixel)

while True:
    try:
        print(header['CROTA2'])
        break
    except:
        print('Value not found')
    break

#This is the coordinate system and projection.

while True:
    try:
        print(header['CTYPE1'], header['CTYPE2'])
        break
    except:
        print('Value not found')
    break
#We are now going to extract and pring out the 'Telescop' value from the header.
#Then we will extract the wavelnth and waveunit values to construct an astropy quantity object
#for the wavelenght of the image.

while True:
    try:
        print(header['TELESCOP'])
        break
    except:
        print('Value not found')
    break

while True:
    try:
        quantity = u.Quantity(header['WAVELNTH'], unit=header['WAVEUNIT'])
        print(quantity)
        break
    except:
        print('Value not found')
    break

print(a)
sourcename = hdu_list[0].data
print(type(sourcename))
print('Shape:',sourcename.shape)
image_data = sourcename
histogram1()
plt.title('Source Image')
plt.show()

target = input('Enter the name of the Reference image:')
hdu_list1 = fits.open(target)
b = hdu_list1.info()
print(b)
targetname = hdu_list1[0].data
print(type(targetname))
print('Shape:',targetname.shape)
image_data = targetname

histogram1()
plt.title('Target Image')
plt.show()

image_data = targetname
blob_finder3(image_data)

if __name__ == '__main__':
    image_gray = image_data
    question = input('Has any error occured? If so, answer YES, if not answer NO. , as well, if you are using an image with a subtracted background, answer DO NOT APPLY to this question, so the contrast enhancement will not be applied')
    b = int(input('Choose the brightness of the image between 1-100. The lower the number, the darker the image and viceversa'))

    if question == 'YES':
        contingency = image_gray * np.log10(image_gray, where=image_gray > 0, casting='unsafe')
        ax.imshow(contingency, origin='lower',
        norm=LogNorm(vmin=np.min(contingency)+1, vmax = np.max(contingency)*(1/b)),
        cmap='gray')
    if question == 'NO':
        normalcy = image_gray * np.log10(image_gray)
        ax.imshow(normalcy, origin='lower',
        norm=LogNorm(vmin=np.min(normalcy), vmax=np.max(normalcy)*(1/b)),
        cmap='gray')
    elif question == 'DO NOT APPLY':
        ax.imshow(image_gray, cmap='gray')

    cv2.setMouseCallback('Astronomical Image', click_event(event, x, y, flags, params))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
