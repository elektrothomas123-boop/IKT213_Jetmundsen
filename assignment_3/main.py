import cv2
import numpy as np
import os


def sobel_edge_detection(image):
    """
    Detect edges using Sobel filter

    Args:
        image: Input image (BGR format)

    Returns:
        sobel_combined: Sobel edge detected image
    """
    # Convert to grayscale if the image is colored
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Gaussian blur with ksize=(3,3) and sigmaX=0
    blurred = cv2.GaussianBlur(gray, (3, 3), sigmaX=0)

    # Apply Sobel edge detection with dx=1, dy=1, ksize=1
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, dx=1, dy=0, ksize=1)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, dx=0, dy=1, ksize=1)

    # Combine both gradients
    sobel_combined = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Convert to uint8 format for saving
    sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))

    # Save the image
    cv2.imwrite('sobel_edge_detection.png', sobel_combined)

    return sobel_combined


def canny_edge_detection(image, threshold_1, threshold_2):
    """
    Detect edges using Canny filter

    Args:
        image: Input image (BGR format)
        threshold_1: First threshold for the hysteresis procedure
        threshold_2: Second threshold for the hysteresis procedure

    Returns:
        canny_edges: Canny edge detected image
    """
    # Convert to grayscale if the image is colored
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Gaussian blur with ksize=(3,3) and sigmaX=0
    blurred = cv2.GaussianBlur(gray, (3, 3), sigmaX=0)

    # Apply Canny edge detection
    canny_edges = cv2.Canny(blurred, threshold_1, threshold_2)

    # Save the image
    cv2.imwrite('canny_edge_detection.png', canny_edges)

    return canny_edges


def main():
    """
    Main function to run the edge detection algorithms
    """
    # Load the image using cv2.imread()
    image = cv2.imread('lambo.png')

    if image is None:
        print("Error: Could not load image 'lambo.png'. Please make sure the file exists in the current directory.")
        return

    print("Image loaded successfully!")
    print(f"Image shape: {image.shape}")

    # Apply Sobel edge detection
    print("Applying Sobel edge detection...")
    sobel_result = sobel_edge_detection(image)
    print("Sobel edge detection completed and saved as 'sobel_edge_detection.png'")

    # Apply Canny edge detection with threshold_1=50 and threshold_2=50
    print("Applying Canny edge detection...")
    canny_result = canny_edge_detection(image, threshold_1=50, threshold_2=50)
    print("Canny edge detection completed and saved as 'canny_edge_detection.png'")

    # Display results (optional - for testing purposes)
    cv2.imshow('Original Image', image)
    cv2.imshow('Sobel Edge Detection', sobel_result)
    cv2.imshow('Canny Edge Detection', canny_result)

    print("Press any key to close the windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()