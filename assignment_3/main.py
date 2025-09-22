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


def template_match(image, template):
    """
    Perform template matching to find template in image

    Args:
        image: Input image (BGR format)
        template: Template image (BGR format)

    Returns:
        result_image: Image with matched areas marked with red rectangles
    """
    # Convert both images to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Get template dimensions
    h, w = gray_template.shape

    # Perform template matching
    result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)

    # Set threshold
    threshold = 0.9

    # Find locations where matching exceeds threshold
    locations = np.where(result >= threshold)

    # Create a copy of the original image to draw rectangles
    result_image = image.copy()

    # Draw red rectangles around matched areas
    for pt in zip(*locations[::-1]):  # Switch columns and rows
        cv2.rectangle(result_image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    # Save the result
    cv2.imwrite('template_match_result.png', result_image)

    return result_image


def resize(image, scale_factor, up_or_down):
    """
    Resize image using image pyramids

    Args:
        image: Input image
        scale_factor: Integer scale factor (should be 2)
        up_or_down: String "up" or "down"

    Returns:
        resized_image: Resized image
    """
    resized_image = image.copy()

    # Apply scaling multiple times based on scale_factor
    for i in range(scale_factor):
        if up_or_down.lower() == "up":
            # Scale up using pyrUp
            resized_image = cv2.pyrUp(resized_image)
        elif up_or_down.lower() == "down":
            # Scale down using pyrDown
            resized_image = cv2.pyrDown(resized_image)
        else:
            print(f"Invalid direction: {up_or_down}. Use 'up' or 'down'")
            return image

    # Save the resized image
    filename = f'resized_{up_or_down}_{scale_factor}.png'
    cv2.imwrite(filename, resized_image)

    return resized_image


def main():
    """
    Main function to run all image processing algorithms
    """
    # Part 1: Sobel and Canny edge detection on lambo.png
    print("=== PART 1: Edge Detection ===")
    lambo_image = cv2.imread('lambo.png')

    if lambo_image is None:
        print("Error: Could not load image 'lambo.png'. Please make sure the file exists in the current directory.")
        return

    print("Lambo image loaded successfully!")
    print(f"Lambo image shape: {lambo_image.shape}")

    # Apply Sobel edge detection
    print("Applying Sobel edge detection...")
    sobel_result = sobel_edge_detection(lambo_image)
    print("Sobel edge detection completed and saved as 'sobel_edge_detection.png'")

    # Apply Canny edge detection
    print("Applying Canny edge detection...")
    canny_result = canny_edge_detection(lambo_image, threshold_1=50, threshold_2=50)
    print("Canny edge detection completed and saved as 'canny_edge_detection.png'")

    # Part 2: Template matching
    print("\n=== PART 2: Template Matching ===")
    shapes_image = cv2.imread('shapes-1.png')
    template_image = cv2.imread('shapes_template.jpg')

    if shapes_image is None:
        print("Error: Could not load 'shapes-1.png'. Please make sure the file exists.")
    elif template_image is None:
        print("Error: Could not load 'shapes_template.jpg'. Please make sure the file exists.")
    else:
        print("Template matching images loaded successfully!")
        print("Applying template matching...")
        template_result = template_match(shapes_image, template_image)
        print("Template matching completed and saved as 'template_match_result.png'")

    # Part 3: Resizing
    print("\n=== PART 3: Image Resizing ===")
    print("Applying image resizing...")

    # Resize up by factor of 2
    resize_up_result = resize(lambo_image, scale_factor=2, up_or_down="up")
    print("Image resized UP and saved as 'resized_up_2.png'")

    # Resize down by factor of 2
    resize_down_result = resize(lambo_image, scale_factor=2, up_or_down="down")
    print("Image resized DOWN and saved as 'resized_down_2.png'")

    # Display results (optional - for testing purposes)
    print("\nDisplaying results...")
    cv2.imshow('Original Lambo', lambo_image)
    cv2.imshow('Sobel Edge Detection', sobel_result)
    cv2.imshow('Canny Edge Detection', canny_result)

    if 'template_result' in locals():
        cv2.imshow('Template Matching Result', template_result)

    cv2.imshow('Resized Up', resize_up_result)
    cv2.imshow('Resized Down', resize_down_result)

    print("Press any key to close all windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\nAll tasks completed successfully!")
    print("Generated files:")
    print("- sobel_edge_detection.png")
    print("- canny_edge_detection.png")
    if 'template_result' in locals():
        print("- template_match_result.png")
    print("- resized_up_2.png")
    print("- resized_down_2.png")


if __name__ == "__main__":
    main()