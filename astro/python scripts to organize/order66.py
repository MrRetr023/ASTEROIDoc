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

#Set directory strucuture, make sure to check this.
# Set directory structure
cwd = os.getcwd()
proc_dir = os.path.join(cwd, 'processed')
data_dir = os.path.join(cwd, 'data')
out_dir = os.path.join(proc_dir, 'out')
if os.path.isdir(proc_dir):
    shutil.rmtree(proc_dir)
os.mkdir(proc_dir)

for f in os.listdir(data_dir):
    shutil.copy2(os.path.join(data_dir, f), os.path.join(proc_dir, f))
os.chdir(proc_dir)

def histogram1():
    img_stretch = image_data * np.log10(image_data)
    histogram = plt.hist(np.concatenate(img_stretch), bins=100*np.linspace(0,256), log=True)
    norm = ImageNormalize(img_stretch, interval=MinMaxInterval() ,stretch=SqrtStretch())
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(img_stretch, origin = 'lower',
        norm=LogNorm(vmin = np.min(img_stretch), vmax = np.max(img_stretch)*(1/45)),
        cmap="gray") #grey/binary
    fig.colorbar(im)
    return im
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

target_image_aligned_name=os.path.join(proc_dir, "bg_sub_target.fits")
registered_image_aligned_name=os.path.join(proc_dir, "bg_sub_registered.fits")

source = input('Enter the name of the Science image:')
hdu_list = fits.open(source)
a = hdu_list.info()
print(a)
sourcename = hdu_list[0].data
print(type(sourcename))
print('Shape:',sourcename.shape)
image_data = sourcename
histogram1()
plt.axis('off')
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
plt.axis('off')
plt.title('Target Image')
plt.show()

sourcelist = sourcename
targetlist = targetname
#I need to add an alternative for troubleshooting
transf, (sourcetransf, targettransf) = aa.find_transform(sourcelist, targetlist)

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

#Displaying absoluteregistered
img_data = absoluteregistered
norm = ImageNormalize(img_data, interval=MinMaxInterval() ,stretch=SqrtStretch())
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
im = ax.imshow(img_data, origin = 'lower', cmap="gray") #grey/binary
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
hdu_image_sub.writeto(registered_image_aligned_name)
#REMOVE THE BACKGROUND FROM THE Target Image
targetlist = targetlist - bkg_target.background
hdu_image_sub = fits.PrimaryHDU(targetlist-bkg_target.background)
hdu_image_sub.writeto(target_image_aligned_name)

#Display the Background subtraction
backgroundregistered = bkg_registered.background
plt.imshow(bkg_registered.background, origin='lower', cmap='Greys_r')
plt.title('Background of the aligned image')
plt.show()

backgroundtarget = bkg_target.background
plt.imshow(backgroundtarget, origin='lower', cmap='Greys_r')
plt.title('Background of the target image')
plt.show()

#Motion detecion attempt using SAD in cv2, THIS NEEDS REVISION

registered_image_int = registered_image1.astype('uint8')
#frame = imutils.resize(registered_image_int)
#gray = cv2.cvtColor(registered_image_int, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(registered_image_int, (21, 21), 0)
average = (np.average(targetname) / np.average(registered_image))

targetlist1 = targetlist.astype('uint8')
#targetgray = cv2.GaussianBlur(targetlist1, (21, 21), 0)
registered_image2 = registered_image1.astype('uint8')
framedelta = cv2.absdiff(targetlist1, gray)
framedelta1 = cv2.bitwise_and(framedelta, framedelta)
threshold = cv2.threshold(framedelta, 180, 255, cv2.THRESH_BINARY_INV) [1]

#consider the next function as a possible utility
#threshold = cv2.dilate(threshold, None, iterations = 2)

#THIS NEEDS REVISION

registeredabs = registered_image1.astype('uint8')
targetabs = targetname.astype('uint8')
image = cv2.absdiff(registeredabs, targetabs) + threshold#registered_image1 - targetlist
#change threshold for framedelta
image_data = threshold
#img_stretch = image_data * np.log10(image_data, out=image_data, where=image_data > 0, casting="unsafe")
#histogram = plt.hist(np.concatenate(img_stretch), bins=100*np.linspace(0,256), log=True)
#norm = ImageNormalize(img_stretch, interval=MinMaxInterval() ,stretch=SqrtStretch())
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
im = ax.imshow(image_data, origin = 'lower', cmap="binary")
    #norm=LogNorm(vmin = np.min(image_data)+1, vmax = np.max(image_data)*(1)),
    #cmap="gray") #grey/binary
fig.colorbar(im)
plt.title('Sample output 1, cv2, motion detection')
plt.show()

image_data = image
#img_stretch = image_data * np.log10(image_data, out=image_data, where=image_data > 0, casting="unsafe")
#histogram = plt.hist(np.concatenate(img_stretch), bins=100*np.linspace(0,256), log=True)
#norm = ImageNormalize(img_stretch, interval=MinMaxInterval() ,stretch=SqrtStretch())
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
