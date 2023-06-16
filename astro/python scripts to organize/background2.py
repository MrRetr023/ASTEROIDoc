import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

def energyfunction(a, b, c, x):
    return a*x**2+b*x+c

a = -4.85725*(10**-8)
b = 0.244271
c = -0.2695

df = pd.read_csv('NMTBckgrd3day.csv')
# apply energy function to all values in DataFrame
original_values = df.iloc[:, 0]
new_values = energyfunction(a, b, c, original_values)
df.iloc[:, 0] = new_values
# remove any rows with NaN values
df = df.dropna()
# extract energy and count data
energy = df.iloc[:, 0]
counts = df.iloc[:, 1:]
print(counts)
# normalize counts to peak value

plt.plot(energy, counts)
plt.xlabel('Energy (keV)')
plt.ylabel('Count Rate)')
plt.title('Background counts')
plt.show()
