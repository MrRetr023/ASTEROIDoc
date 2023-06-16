import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Define the Julia set function
def julia_set(z, c):
    return z ** 2 + c

# Define the neural network to approximate the Julia set function
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2)
])

# Load the trained weights of the neural network
model.load_weights('julia_set_weights.h5')

# Define the resolution of the plot
n = 500
x_min, x_max, y_min, y_max = -2, 2, -2, 2

# Create a grid of complex numbers
x = np.linspace(x_min, x_max, n)
y = np.linspace(y_min, y_max, n)
z = np.meshgrid(x, y)
c = np.zeros((n, n)) + 0.8j

# Flatten the grid of complex numbers
z_flat = np.vstack((z[0].flatten(), z[1].flatten())).T

# Use the neural network to compute the Julia set
julia_set_flat = model.predict([z_flat, c.flatten()[:, None]])
julia_set = julia_set_flat[:, 0] + julia_set_flat[:, 1] * 1j

# Reshape the Julia set into a 2D array
julia_set = julia_set.reshape((n, n))

# Plot the Julia set
plt.imshow(np.abs(julia_set), cmap='jet', extent=[x_min, x_max, y_min, y_max])
plt.axis('off')
plt.show()
