import cv2
import numpy as np

# Read the image
image = cv2.imread("/Users/jakobstrozberg/Documents/GitHub/AER850_Project_3/motherboard_image2.JPEG")

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Thresholding the image to create a binary image
# You might need to adjust the threshold value based on your image
_, binary = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour, assuming it's the motherboard
largest_contour = max(contours, key=cv2.contourArea)

# Create an all-white mask
mask = np.zeros_like(image)

# Fill the mask with white color at the largest contour (which is the motherboard)
cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

# Convert the mask to grayscale
mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

# Create the final image by combining the original image with the mask
result = cv2.bitwise_and(image, image, mask=mask_gray)

# Create a uniform black background
black_background = np.zeros_like(image)

# Place the motherboard on the black background
final_result = np.where(result > 0, result, black_background)

# Display the result
cv2.imshow('Motherboard', final_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('motherboard_no_background.png', final_result)