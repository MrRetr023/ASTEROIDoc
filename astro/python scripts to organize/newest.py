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
from matplotlib.patches import Ellipse
import skimage.morphology as morph
import skimage.exposure as skie
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import matplotlib.backend_bases

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
        #image_data2 = np.where(image_data > 0.0000000001, -10)
        img_stretch = image_data * np.log10(image_data, where=image_data > 0, casting = 'unsafe')# out=image_data2, where=image_data2 > 0, casting = 'unsafe')
        histogram = plt.hist(np.concatenate(img_stretch), bins=100*np.linspace(0,256), log=True)
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
        fig = plt.figure()
        wcs = WCS(hdu_list[0].header)
        ax = fig.add_subplot(111, projection = wcs)
        im = ax.imshow(img_stretch, origin = 'lower',
            norm=LogNorm(vmin = np.min(img_stretch), vmax = np.max(img_stretch)*(1/b)),
            cmap="gray") #grey/binary
        fig.colorbar(im)
    return im


def blob_finder3(IMAGE_DATA):
    question = input('Has any error occured? If so, answer YES, if not answer NO. , as well, if you are using an image with a subtracted background, answer DO NOT APPLY to this question, so the contrast enhancement will not be applied')
    b = int(input('Choose the brightness of the image between 1-100. The lower the number, the darker the image and viceversa'))

    image_gray = IMAGE_DATA

    blobs_doh = blob_doh(
        image_gray,
        min_sigma=30,
        max_sigma=90,
        num_sigma=5,
        threshold=40,
        overlap=0.25)

    blobs_list = [blobs_doh]
    colors = ["red"]
    titles = ['Determinant of Hessian']
    sequence = zip(blobs_list, colors, titles)
    fig, ax = plt.subplots()

    for idx, (blobs, color, title) in enumerate(sequence):
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

        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax.add_patch(c)
        ax.set_axis_off()

    plt.show()

def click_event(event, x, y, flags, param):
    #checking for left mouse clicks
    event = None
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

coords = []

def mouse_event():
    def onclick(event):
        global ix, iy
        ix, iy = event.xdata, event.ydata
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
        global coords
        coords.append((ix, iy))
        return coords
    image_gray = registered_image1
    question = input('Has any error occured? If so, answer YES, if not answer NO. , as well, if you are using an image with a subtracted background, answer DO NOT APPLY to this question, so the contrast enhancement will not be applied')
    b = int(input('Choose the brightness of the image between 1-100. The lower the number, the darker the image and viceversa'))
    #wcs = WCS(hdu_list[0].header)
    hdu_list = hdu_list1
    wcs = WCS(hdu_list[0].header)
    fig, ax = plt.subplots(1, 1)
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
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    #fig.canvas.mpl_disconnect(cid)

def blob_findermouseevent(IMAGE_DATA):
    def onclick(event):
        global ix, iy
        ix, iy = event.xdata, event.ydata
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
        global coords
        coords.append((ix, iy))
        return coords

    question = input('Has any error occured? If so, answer YES, if not answer NO. , as well, if you are using an image with a subtracted background, answer DO NOT APPLY to this question, so the contrast enhancement will not be applied')
    b = int(input('Choose the brightness of the image between 1-100. The lower the number, the darker the image and viceversa'))
    IMAGE_DATA = image_data
    image_gray = IMAGE_DATA

    blobs_doh = blob_doh(
        image_gray,
        min_sigma=30,
        max_sigma=90,
        num_sigma=5,
        threshold=40,
        overlap=0.25)

    blobs_list = [blobs_doh]
    colors = ["red"]
    titles = ['Determinant of Hessian']
    sequence = zip(blobs_list, colors, titles)
    fig, ax = plt.subplots()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    for idx, (blobs, color, title) in enumerate(sequence):
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

        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax.add_patch(c)
        ax.set_axis_off()
    plt.show()

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
a = hdu_list1.info()
print(a)
b = hdu_list1[0].header
targetname = hdu_list1[0].data

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
plt.show()

#Substracting absolute values
substraction = registered_image - targetlist
absoluteregistered = np.abs(substraction)

img_data = absoluteregistered
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
im = ax.imshow(img_data, origin = 'lower',
    cmap="gray") #grey/binary
fig.colorbar(im)
plt.title('First outcome')
plt.show()

#Background subtraction

#let us consider the target image aligned and use the registered image as the second part

sigma_clip = SigmaClip(sigma=3, maxiters=10)
bkg_estimator = MedianBackground()
bkg_registered = Background2D(registered_image, (50, 50),
                filter_size = (3,3), sigma_clip = sigma_clip,
                bkg_estimator=bkg_estimator, exclude_percentile=5)
bkg_target = Background2D(targetlist, (50, 50),
                filter_size = (3,3), sigma_clip = sigma_clip,
                bkg_estimator=bkg_estimator, exclude_percentile=5)

#REMOVE THE BACKGROUND FROM THE REGISTERED Image
registered_image11 = registered_image1 - bkg_registered.background
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

#Display the differences between the original and the subtracted-background image

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

#Subtracted-image: Registered image

ax2.imshow(registered_image11, origin='lower', cmap='gray',
           vmin=np.percentile(registered_image11, 1),
           vmax=np.percentile(registered_image11, 99))
ax2.set_title('Background-subtracted image')

#Original image: Registered image
image_data = registered_image1
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
    #image_data2 = np.where(image_data > 0.0000000001, -10)
    img_stretch = image_data * np.log10(image_data, where=image_data > 0, casting = 'unsafe')# out=image_data2, where=image_data2 > 0, casting = 'unsafe')

    if np.min(img_stretch) <= 0:
        vmin = np.min(img_stretch)+1
    elif np.min(img_stretch) > 0:
        vmin = np.min(img_stretch)
    ax1.imshow(img_stretch, origin='lower',
        norm = LogNorm(vmin, vmax=np.max(img_stretch)*(1/b)),
        cmap='gray')
    ax1.set_title('Original Image')
else:
    img_stretch = image_data * np.log10(image_data)
    ax1.imshow(img_stretch, origin = 'lower',
        norm=LogNorm(vmin = np.min(img_stretch), vmax = np.max(img_stretch)*(1/b)),
        cmap="gray") #grey/binary
    ax1.set_title('Original Image')

plt.show()

#Display the differences between the original and the subtracted-background image

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

#Subtracted-image: Target image
ax2.imshow(targetlist2, origin='lower', cmap='gray',
           vmin=np.percentile(targetlist2, 1),
           vmax=np.percentile(targetlist2, 99))
ax2.set_title('Background-subtracted image: Target image')

#Original image: Registered image
image_data = targetlist2
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
    #image_data2 = np.where(image_data > 0.0000000001, -10)
    img_stretch = image_data * np.log10(image_data, where=image_data > 0, casting = 'unsafe')# out=image_data2, where=image_data2 > 0, casting = 'unsafe')

    if np.min(img_stretch) <= 0:
        vmin = np.min(img_stretch)+1
    elif np.min(img_stretch) > 0:
        vmin = np.min(img_stretch)
    ax1.imshow(img_stretch, origin='lower',
        norm = LogNorm(vmin, vmax=np.max(img_stretch)*(1/b)),
        cmap='gray')
    ax1.set_title('Original Image: Target image')
else:
    img_stretch = image_data * np.log10(image_data)
    ax1.imshow(img_stretch, origin = 'lower',
        norm=LogNorm(vmin = np.min(img_stretch), vmax = np.max(img_stretch)*(1/b)),
        cmap="gray") #grey/binary
    ax1.set_title('Original Image: Target image')

plt.show()

#In this section we will estimate the efficiency of the background subtraction.

residualregistered = registered_image1 - registered_image11
residualtarget = targetlist - targetlist2

#We will start by calculating the mean and standard deviation of the pixel values for each image

mean_registered = np.mean(residualregistered)
mean_target = np.mean(residualtarget)

stdev_registered = np.std(residualregistered)
stdev_target = np.std(residualtarget)

#Then we will estimate the RMS noise level from a noise region of the image.

rms_region_registered = residualregistered[100:200, 100:200]
rms_region_target = residualtarget[100:200, 100:200]

rms_registered = np.sqrt(np.mean(np.square(rms_region_registered - mean_registered)))
rms_target = np.sqrt(np.mean(np.square(rms_region_target - mean_target)))

#Then we will compare the mean and standard deviation to the expected values for Gaussian Noise.

print('Mean for the registered image:', mean_registered)
print('Mean for the target image:', mean_target)

print('Expected mean for the registered image:', 0)
print('Expected mean for the target image:', 0)

print('Standard deviation for the registered image:', stdev_registered)
print('Standard deviation for the target image:', stdev_target)

print('Expected standard deviation for the registered image:', rms_registered)
print('Expected standard deviation for the target image:', rms_target)

#At this point we will also perform object detection.

registered_image3 = registered_image11.astype('uint8')
targetlist5 = targetlist2.astype('uint8')
gray = cv2.GaussianBlur(registered_image3, (13, 13), 0)
targetlist6 = cv2.GaussianBlur(targetlist5, (13, 13), 0)
diff_frame1 = cv2.absdiff(targetlist6, registered_image3)
diff_frame = cv2.dilate(diff_frame1, None, iterations = 1)#11)#, kernel, 1)
thresh_frame = cv2.threshold(diff_frame, 200, 255, type=cv2.THRESH_BINARY_INV)[1]
#Second outcome
registeredabs = registered_image11.astype('uint8')
targetabs = targetname.astype('uint8')
image = cv2.absdiff(registeredabs, targetabs) - thresh_frame

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
im = ax.imshow(image_data, origin = 'lower', cmap="gray")
    #norm=LogNorm(vmin = np.min(image_data)+1, vmax = np.max(image_data)*(1)),
    #cmap="gray") #grey/binary
fig.colorbar(im)
plt.title('Second outcome')
plt.show()

#Third outcome

image_data = thresh_frame
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
im = ax.imshow(image_data, origin = 'lower', cmap="binary")
    #norm=LogNorm(vmin = np.min(image_data)+1, vmax = np.max(image_data)*(1)),
    #cmap="gray") #grey/binary
fig.colorbar(im)
plt.title('Third outcome, motion detection')
plt.show()

#Fourth outcome

registeredabs = registered_image11.astype('uint8')
targetabs = targetname.astype('uint8')
image = cv2.absdiff(registeredabs, targetabs) - thresh_frame#threshold#registered_image1 - targetlist

#change threshold for framedelta
image_data = cv2.absdiff(registeredabs, targetabs) - thresh_frame
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
im = ax.imshow(image_data, origin = 'lower', cmap="binary")
    #norm=LogNorm(vmin = np.min(image_data)+1, vmax = np.max(image_data)*(1)),
    #cmap="gray") #grey/binary
fig.colorbar(im)
plt.title('Fourth outcome')
plt.show()

#Cleaned up image.

imageframe = thresh_frame - cv2.absdiff(registeredabs, targetabs)
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

#Object detection applying the blob function with stored coordinates

registered_imagesep = cv2.absdiff(registered_image11, bkg_registered.background)
image_data = registered_image1
blob_findermouseevent(image_data)
plt.show()
print('PASS')
print(coords)
