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

sourcelist = sourcename
targetlist = targetname

while True:
    try:
        transf, (sourcetransf, targettransf) = aa.find_transform(sourcelist, targetlist)
        break
    except aa.MaxIterError:
        while True:
            try:
                tranfs, (sourcetransf, targettransf) = aa.find_transform(sourcelist, targetlist, detection_sigma=2)
                break
            except:
                while True:
                    try:
                        transf, (sourcetransf, targettransf) = aa.find_transform(sourcelist, targetlist, max_control_points=3)
                        break
                    except:
                        #we will attempt to reduce the background and then clean the picture more before applying the transform
                        data = targetlist.astype(np.uint16)
                        footprint = circular_footprint(radius=3)
                        segm = SegmentationImage(data)
                        masktarget = segm.make_source_mask(footprint = footprint)
                        data1 = sourcelist
                        segm1 = SegmentationImage(data1)
                        masksource = segm1.make_source_mask(footprint = footprint)

                        sigma_clip = SigmaClip(sigma=3, maxiters=10)
                        bkg_estimator = MedianBackground()
                        bkg_target = Background2D(targetlist, (200, 150), filter_size = (3,3), sigma_clip = sigma_clip, bkg_estimator = bkg_estimator)
                        bkg_source = Background2D(sourcelist, (200, 150), filter_size = (3,3), sigma_clip = sigma_clip, bkg_estimator = bkg_estimator)

                        #Removing the background

                        targetlist1 = targetlist - bkg_target.background
                        sourcelist1 = sourcelist - bkg_source.background

                        targetlist2 = targetlist1.astype('uint8')
                        sourcelist2 = sourcelist1.astype('uint8')

                        targetlist3 = cv2.GaussianBlur(targetlist2, (13,13), 0)
                        sourcelist3 = cv2.GaussianBlur(sourcelist2, (13,13), 0)
                        while True:
                            try:
                                transf, (sourcetransf, targettransf) = aa.find_transform(sourcelist3, targetlist3)
                                break
                            except:
                                while True:
                                    try:
                                        tranfs, (sourcetransf, targettransf) = aa.find_transform(sourcelist3, targetlist3, detection_sigma=2)
                                        break
                                    except:
                                        while True:
                                            try:
                                                transf, (sourcetransf, targettransf) = aa.find_transform(sourcelist3, targetlist3, max_control_points=3)
                                                break
                                            except:
                                                print('Your image does not have enough reference stars to perform the alignment, please select another image')
                                    break
                            break
                    break
            break
        print('Your image does not have enough reference stars to perform the alignment, please select another image')
    break

#These are the parameters of the transform.

print('Rotation:{:.2f} degrees'.format(transf.rotation *180.0 / np.pi))
print('\nScale factor: {:.2f}'.format(transf.scale))
print('\nTranslation: (x, y) = ({:.2f}, {:.2f})'.format(*transf.translation))
print("\nTranformation matrix:\n{}".format(transf.params))
print("\nPoint correspondence:")
for (x1, y1), (x2, y2) in zip(sourcetransf, targettransf):
    print("({:.2f}, {:.2f}) is source --> ({:.2f}, {:.2f}) in target"
          .format(x1, y1, x2, y2))

registered_image = aa.apply_transform(transf, sourcelist, targetlist, fill_value=None, propagate_mask=False)[0]
registered_image1 = registered_image
image_data = registered_image1

histogram1()
plt.axis('off')
plt.show()

#Substracting absolute values
substraction = registered_image - targetlist
absoluteregistered = np.abs(substraction)

print('PAY ATTENTION', absoluteregistered)
img_data = absoluteregistered

norm = ImageNormalize(img_data, interval=MinMaxInterval() ,stretch=SqrtStretch())
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
im = ax.imshow(img_data, origin = 'lower',
    #norm=LogNorm(vmin=np.min(img_data)+1, vmax=np.max(img_data)*(1)),
    cmap="gray") #grey/binary
fig.colorbar(im)
plt.title('Change without removing background and LogNorm, KEEP!')
plt.show()

#Background subtraction

footprint = circular_footprint(radius=3)
data = registered_image.astype(np.uint16)
segm = SegmentationImage(data)
maskregistered = segm.make_source_mask(footprint = footprint)
data1 = targetlist
segm1 = SegmentationImage(data1)
masktarget = segm1.make_source_mask(footprint = footprint)

#let us consider the target image aligned and use the registered image as the second part

sigma_clip = SigmaClip(sigma=3, maxiters=10)
bkg_estimator = MedianBackground()
bkg_registered = Background2D(registered_image, (200, 150), filter_size = (3,3), sigma_clip = sigma_clip, bkg_estimator=bkg_estimator)
bkg_target = Background2D(targetlist, (200, 150), filter_size = (3,3), sigma_clip = sigma_clip, bkg_estimator=bkg_estimator)

#REMOVE THE BACKGROUND FROM THE REGISTERED Image
registered_image1 = registered_image1 - bkg_registered.background
hdu_image_sub = fits.PrimaryHDU(registered_image1-bkg_registered.background)
#hdu_image_sub.writeto(registered_image_aligned_name)
#REMOVE THE BACKGROUND FROM THE Target Image
targetlist2 = targetlist - bkg_target.background
hdu_image_sub = fits.PrimaryHDU(targetlist-bkg_target.background)
#hdu_image_sub.writeto(target_image_aligned_name)

#Display the Background subtraction
backgroundregistered = bkg_registered.background
plt.imshow(bkg_registered.background, origin='lower', cmap='Greys_r')
plt.title('Background of the aligned image')
plt.show()

backgroundtarget = bkg_target.background
plt.imshow(backgroundtarget, origin='lower', cmap='Greys_r')
plt.title('Background of the target image')
plt.show()

#Image subtraction only possible if using the Sextractor, so it will be avoided in the meantime by using the method shown in GROWTH project

#Motion detecion attempt using SAD in cv2
#Motion detection without taking off the background
registered_image3 = registered_image1.astype('uint8')
targetlist5 = targetlist2.astype('uint8')
gray = cv2.GaussianBlur(registered_image3, (13, 13), 0)
targetlist6 = cv2.GaussianBlur(targetlist5, (13, 13), 0)
diff_frame1 = cv2.absdiff(targetlist6, gray) #registered_image3)
kernel = Gaussian2DKernel(x_stddev=1)
kernel1 = np.array(kernel)
diff_frame = cv2.dilate(diff_frame1, None, iterations = 1)#11)#, kernel, 1)
thresh_frame = cv2.threshold(diff_frame, 200, 255, type=cv2.THRESH_BINARY_INV)[1]

registered_image_int = registered_image1.astype('uint8')
frame = imutils.resize(registered_image_int)

gray = cv2.GaussianBlur(registered_image_int, (21, 21), 0)

targetlist1 = targetlist.astype('uint8')
registered_image2 = registered_image1.astype('uint8')
framedelta = cv2.absdiff(targetlist1, gray)

threshold = cv2.threshold(framedelta, 180, 255, cv2.THRESH_BINARY_INV) [1]

registeredabs = registered_image1.astype('uint8')
targetabs = targetname.astype('uint8')
image = cv2.absdiff(registeredabs, targetabs) - thresh_frame#threshold#registered_image1 - targetlist

#change threshold for framedelta
image_data = thresh_frame
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
im = ax.imshow(image_data, origin = 'lower', cmap="binary")
    #norm=LogNorm(vmin = np.min(image_data)+1, vmax = np.max(image_data)*(1)),
    #cmap="gray") #grey/binary
fig.colorbar(im)
plt.title('Sample output 1, cv2, motion detection')
plt.show()

image_data = image

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
im = ax.imshow(image_data, origin = 'lower', cmap="gray")
    #norm=LogNorm(vmin = np.min(image_data)+1, vmax = np.max(image_data)*(1)),
    #cmap="gray") #grey/binary
fig.colorbar(im)
plt.title('Sample output 2, sum, this one')
plt.show()

imageframe = threshold - cv2.absdiff(registeredabs, targetabs)
img_data = np.abs(backgroundregistered - registered_image)
img_stretch = image_data * np.log10(image_data, out=image_data, where=image_data > 0, casting="unsafe")
histogram = plt.hist(np.concatenate(img_stretch), bins=100*np.linspace(0,256), log=True)
norm = ImageNormalize(img_stretch, interval=MinMaxInterval() ,stretch=SqrtStretch())
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
im = ax.imshow(img_data, origin = 'lower',
    norm=LogNorm(vmin = np.min(img_data)+1, vmax = np.max(img_data)*(1)),
    cmap="gray") #grey/binary
fig.colorbar(im)
plt.title('Sample output 3, subtraction. npabs')
plt.show()
