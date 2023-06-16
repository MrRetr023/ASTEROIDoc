import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.colors import LogNorm
from astropy.visualization import (MinMaxInterval, SqrtStretch, ImageNormalize)
from astropy.visualization import ImageNormalize
name = input('Enter the name of the file:')
rewrite = input('Do you want to rewrite the changes into a new fit file?')
hdu_list = fits.open(name)
a = hdu_list.info()
print(a)
image_data = hdu_list[0].data
#image_data2 = image_data.astype('float64')

print('Min:', np.min(image_data))
print('Max:', np.max(image_data))
print('Mean:', np.mean(image_data))
print('Stdev:', np.std(image_data))

a = float(np.min(image_data))
b = float(np.max(image_data))
constant = (255-0)/(b-a)

def histogram():
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
histogram()
plt.axis('off')
plt.show()

if rewrite == 'yes':
    with fits.open(name, mode = 'update') as hdul:
        #Change something in hdul.
        hdul.flush() #changes are written back to name.
# closing the file will also flush any changes and prevent further writing
else:
    pass
