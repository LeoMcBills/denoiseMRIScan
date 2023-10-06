import cv2
import numpy as np

# Load your low-field MRI image
input_image = cv2.imread('mri1.jpg', cv2.IMREAD_GRAYSCALE)

# Convert the image to float32 for processing
input_image = np.float32(input_image)

# Perform Total Variation denoising
denoised_image = cv2.ximgproc.denoiseTVL1(input_image, 1.0, 30)

# Convert the denoised image back to uint8 for display
denoised_image = np.uint8(np.clip(denoised_image, 0, 255))

# Save the denoised image
cv2.imwrite('denoised_mri_image.png', denoised_image)

# Display the original and denoised images (optional)
cv2.imshow('Original MRI', input_image)
cv2.imshow('Denoised MRI', denoised_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
