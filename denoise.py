#!/usr/bin/env python3

'''a simple python script demonstrating Total Variation denoising 
using opencv, numpy and scipy'''

import cv2
from scipy.signal import convolve2d
import numpy as np

noisedImage = cv2.imread('mri1.jpg', cv2.IMREAD_GRAYSCALE)
input_image = np.float32(noisedImage)

# Define Total Variation denoising function
def tv_denoise(image, weight=0.1, iterations=30):
    """ perform total-variation denoising on a grayscale image
    using the anisotropic diffusion equation
    partial I / partial t = div( c(x,y,t) grad I )
    where c = 1 / (1 + |grad I|^2/weight), and t is time
    parameters:
        image - input image to be denoised
        weight - (optional) weight parameter for the denoising equation
        iterations - (optional) number of iterations to run
    returns:
        denoised image as a numpy array
    """
    u = image.copy()
    px = np.zeros_like(image)
    py = np.zeros_like(image)
    rho = 1.0

    for _ in range(iterations):
        u_old = u
        '''Calculating the gradient of u using the central difference method'''
        ux = np.roll(u, -1, axis=1) - u
        uy = np.roll(u, -1, axis=0) - u

        ## evaluating edge stopping function
        px_new = px + (1.0 / rho) * ux
        py_new = py + (1.0 / rho) * uy
        norm_new = np.maximum(1, np.sqrt(px_new**2 + py_new**2))
        px = px_new / norm_new
        py = py_new / norm_new

        '''Calculating divergence of the edge stopping function'''
        rx = np.roll(px, 1, axis=1)
        ry = np.roll(py, 1, axis=0)
        div_p = (px - rx) + (py - ry)
        u = image + weight * div_p
    return u

'''
Let me explain the above code a bit. The function tv_denoise() takes in three parameters:
image - the input image to be denoised
weight - (optional) weight parameter for the denoising equation
iterations - (optional) number of iterations to run
The function returns the denoised image as a numpy array.
The first step in the function is to make a copy of the input image.
Then we initialize the variables px and py to zero. These variables are used to store the gradient of the image.
The variable rho is the step size for the gradient descent method.
The for loop runs for the number of iterations specified by the user.
Inside the for loop, we first calculate the gradient of the image using the central difference method.
Then we calculate the edge stopping function. This is the function that determines the direction of the gradient.
The divergence of the edge stopping function is calculated next.
Finally, the denoised image is calculated using the divergence of the edge stopping function.
The denoised image is returned by the function.
'''
denoised_image = tv_denoise(input_image, weight=1.0, iterations=30)

## Conversion back to uint8 for display
denoised_image = np.uint8(np.clip(denoised_image, 0, 255))

## saving and displaying the results
cv2.imwrite('denoised.jpg', denoised_image)
cv2.imshow('Original MRI', input_image)
cv2.imshow('Denoised MRI', denoised_image)
cv2.waitKey(0)
cv2.destroyAllWindows()