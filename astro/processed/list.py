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
def histogram1():
    question = input('Has any error occurred?')
    try:
        img_stretch = image_data * np.log10(image_data)
    except:
        print('An ERROR has occurred, answer YES to the first question.')
    if question == 'YES':
        image_data
        image_data2 = np.where(image_data > 0.0000000001, image_data, -10)
histogram1()
