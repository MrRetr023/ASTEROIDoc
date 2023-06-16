import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils import Background2D, MedianBackground

# Load the FITS file data
with fits.open('M31.fit') as hdul:
    data = hdul[0].data

# Compute the background using Background2D
bg_estimator = MedianBackground()
bg = Background2D(data, (200, 150), filter_size=(5, 5),
                  sigma_clip=None, bkg_estimator=bg_estimator)

# Subtract the background from the data
data_subtracted = data - bg.background

# Visualize the background-subtracted image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Original image
ax1.imshow(bg.background, origin='lower', cmap='gray')
ax1.set_title('Original')

# Background-subtracted image
ax2.imshow(data_subtracted, origin='lower', cmap='gray',
           vmin=np.percentile(data_subtracted, 1),
           vmax=np.percentile(data_subtracted, 99))
ax2.set_title('Background-subtracted')

plt.show()

def background():
    sigma_clip = SigmaClip(sigma=3, maxiters=10)
    bkg_estimator = MedianBackground()
    bkg_registered = Background2D(registered_image, (200, 150), filter_size = (3,3), sigma_clip = sigma_clip, bkg_estimator=bkg_estimator)
