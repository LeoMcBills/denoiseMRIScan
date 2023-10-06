import cv2
import numpy as np
from scipy.signal import convolve2d

# Load your low-field MRI image
input_image = cv2.imread('mri1.jpg', cv2.IMREAD_GRAYSCALE)

# Convert the image to float32 for processing
input_image = np.float32(input_image)

# Define Total Variation denoising function
def tv_denoise(image, weight=0.1, iterations=30):
    u = image.copy()
    px = np.zeros_like(image)
    py = np.zeros_like(image)
    rho = 1.0

    for _ in range(iterations):
        u_old = u

        # Calculate gradient in x and y directions
        ux = np.roll(u, -1, axis=1) - u
        uy = np.roll(u, -1, axis=0) - u

        # Update p values
        px_new = px + (1.0 / rho) * ux
        py_new = py + (1.0 / rho) * uy
        norm_new = np.maximum(1, np.sqrt(px_new**2 + py_new**2))
        px = px_new / norm_new
        py = py_new / norm_new

        # Update u values
        rx = np.roll(px, 1, axis=1)
        ry = np.roll(py, 1, axis=0)
        div_p = (px - rx) + (py - ry)
        u = image + weight * div_p

    return u

# Perform Total Variation denoising
denoised_image = tv_denoise(input_image, weight=1.0, iterations=30)

# Convert the denoised image back to uint8 for display
denoised_image = np.uint8(np.clip(denoised_image, 0, 255))

# Save the denoised image
cv2.imwrite('denoised_mri_image.jpg', denoised_image)

# Display the original and denoised images (optional)
cv2.imshow('Original MRI', input_image)
cv2.imshow('Denoised MRI', denoised_image)
cv2.waitKey(0)
cv2.destroyAllWindows()