import subprocess

libraries = [
    'numpy',
    'matplotlib',
    'astropy',
    'astroalign',
    'Pillow',
    'scipy',
    'photutils',
    'opencv-python',
    'imutils',
    'sep',
    'sunpy',
    'scikit-image'
]

for library in libraries:
    subprocess.check_call(['pip', 'install', library])
