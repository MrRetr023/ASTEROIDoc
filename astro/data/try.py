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
# Set directory structur
name = input('Enter the name of the file:')
hdu_list = fits.open(name)
a = hdu_list.info()
print(a)
image_data = hdu_list[0].data
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
        ax.set_axis_off())
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
histogram2()
plt.show()
