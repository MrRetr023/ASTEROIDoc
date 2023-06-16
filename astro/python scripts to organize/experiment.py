from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import make_source_mask
import numpy as np
from photutils.utils import circular_footprint
from photutils.segmentation import SegmentationImage
import cv2


image = input('Enter the name of the file:')
hdu_list = fits.open(image)
image_data = hdu_list[0].data

sigma_clip = SigmaClip(sigma=3, maxiters=10) #sigma clipping.
bkg_estimator = MedianBackground()
background = Background2D(image_data, (50, 50), filter_size=(3,3), mask=None, sigma_clip = sigma_clip, bkg_estimator=bkg_estimator)

image = image_data - background.background
gray = cv2.GaussianBlur(image, (13, 13), 0)
diff_frame = cv2.dilate(gray, None, iterations=1)
thresh_frame = cv2.threshold(diff_frame, 200, 255, type=cv2.THRESH_BINARY_INV)[1]

image_data = thresh_frame
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
im = ax.imshow(image_data, origin = 'lower', cmap="binary")
fig.colorbar(im)
plt.show()
