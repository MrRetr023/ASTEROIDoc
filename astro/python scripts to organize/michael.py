from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import make_source_mask
import numpy as np
from photutils.utils import circular_footprint
from photutils.segmentation import SegmentationImage

sourcefits = fits.open('Pallas1.fit')
#targetimage = fits.open('Pallas2.fit')
image_data = sourcefits

datasource = sourcefits[0].data
source_data = datasource
source_stretch = source_data * np.log10(source_data)
fig = plt.figure()
ax = fig.add_subplot(111)
sourceim =  ax.imshow(source_stretch, origin = 'lower',
        norm = LogNorm(vmin = np.min(source_stretch), vmax = np.max(source_stretch)*(1/45)),
        cmap = 'gray')

sigma_clip = SigmaClip(sigma = 3, maxiters = 10)
bkg_estimator = MedianBackground()
background = Background2D (datasource, (10,10), filter_size=(3,3), sigma_clip = sigma_clip, bkg_estimator = bkg_estimator)
totalimage = source_data - background.background
im =  ax.imshow(background.background, origin = 'lower', cmap="gray")# vmin=np.percentile(totalimage, 1), vmax=np.percentile(totalimage, 99), cmap = 'gray')
plt.show()
