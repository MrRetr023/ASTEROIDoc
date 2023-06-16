from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import make_source_mask
import numpy as np
from photutils.utils import circular_footprint
from photutils.segmentation import SegmentationImage


#Let's get started with a contrast enhancement.

image = input('Enter the name of the file:')
hdu_list = fits.open(image)
image_data = hdu_list[0].data

#mask

footprint = circular_footprint(3)
data = image_data.astype(np.uint16)
#threshold = detect_threshold(data, snr=3.0)
#segm = detect_sources(data, threshold, npixels=5)
mask = make_source_mask(data, nsigma=9, npixels=5)

#Now we are going to estimate the background.

sigma_clip = SigmaClip(sigma=3, maxiters=10) #sigma clipping.
bkg_estimator = MedianBackground()
background = Background2D(image_data, (50, 50), filter_size=(3,3), mask=mask, sigma_clip = sigma_clip, bkg_estimator=bkg_estimator)

image = image_data - background.background

#Let's see how that looks lke!
fig = plt.figure()
backgroundisplay = background.background
im = plt.imshow(image, origin='lower', cmap='gray',
           vmin=np.percentile(image, 49),
           vmax=np.percentile(image, 50))
fig.colorbar(im)
plt.title('Background of the Image')
plt.show()
