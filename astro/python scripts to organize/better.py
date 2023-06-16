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
from skimage import exposure

def greyscale_stretching(image_data, a, b):
    #This will be performed by using a sigmoid function.
    image_data = 1/(1+np.exp(-a*(image_data-b)))
    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
    plt.imshow(image_data, origin='lower', cmap='gray')
    plt.show()

source = input('Enter the name of the source image:')
sourceimage = fits.open(source)
sourcedata = sourceimage[0].data
image_data = sourcedata
mean = np.mean(image_data)
b = mean 
a=0.02 + 0.01 * np.log10(np.std(image_data))
greyscale_stretching(image_data, a, b)
