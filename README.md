# illumination-baseline

predict illumination from image(s)

This is a baseline model to predict illumination from a single image.

Its network is straighforward:
	input an image (size of 256x256, 3 channels).
	output 3 orders SH coefficients.

This network consist of 7 convolutional layers and 4 fully-connectional layers.
