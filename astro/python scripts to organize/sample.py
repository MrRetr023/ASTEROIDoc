import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

# Load the FITS image into a numpy array
image_data = fits.getdata('123.fit')

# Calculate the mean and standard deviation of the pixel values
mean = np.mean(image_data)
std = np.std(image_data)

# Define the parameters for the sigmoid function
a = 0.02   # controls the slope of the function
b = mean   # controls the midpoint of the function

# Apply the sigmoid function to the image
stretched_data = 1 / (1 + np.exp(-a*(image_data-b)))

# Scale the pixel values to the range [0, 1]
stretched_data = (stretched_data - np.min(stretched_data)) / (np.max(stretched_data) - np.min(stretched_data))

# Display the stretched image
plt.imshow(stretched_data, origin = 'lower', cmap='gray')
plt.show()
