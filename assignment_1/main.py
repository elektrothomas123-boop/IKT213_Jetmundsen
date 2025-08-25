import cv2
import numpy as np
import os
import warnings
import sys

# Suppress macOS camera warnings
warnings.filterwarnings("ignore")

def print_image_information(image):
    """
    Task IV: Print image information as specified in assignment
    Function arguments: image
    Prints: height, width, channels, size (number of values in the cubed array), data type
    """
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    size = image.size
    data_type = image.dtype
    
    print(f"height: {height}")
    print(f"width: {width}") 
    print(f"channels: {channels}")
    print(f"size (number of values in the cubed array): {size}")
    print(f"data type: {data_type}")

def main():
    # Task IV: Print image information for lena.png
    print("=== Image information ===")
    image = cv2.imread('lena.png')
    
    if image is None:
        print("Error: Could not load lena.png. Make sure the file is in the current directory.")
        return
    
    print_image_information(image)
    
    # Task V: Web camera information
    # Suppress stderr to hide AVCapture warning on macOS
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    # Try different camera indices
    cap = None
    camera_index = None
    for idx in [0, 1, 2]:
        test_cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if test_cap.isOpened():
            ret, frame = test_cap.read()
            if ret:
                cap = test_cap
                camera_index = idx
                break
            test_cap.release()
    
    # Restore stderr  
    sys.stderr = old_stderr
    
    if cap is not None and cap.isOpened():
        # Get camera properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print camera information to console
        print("\n=== Camera information ===")
        print(f"fps: {fps}")
        print(f"height: {height}")
        print(f"width: {width}")
        
        # Create solutions directory if it doesn't exist
        os.makedirs('solutions', exist_ok=True)
        
        # Save camera outputs to txt file as specified
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'solutions', 'camera_outputs.txt')
        with open('solutions/camera_outputs.txt', 'w') as f:
            f.write(f"fps: {fps}\n")
            f.write(f"height: {height}\n")
            f.write(f"width: {width}\n")
        
        print(f"\n🟢 V. Web camera information saved into: {output_path} (camera_index={camera_index})")
    else:
        print("\nError: Could not open any camera. Please check camera permissions.")

if __name__ == "__main__":
    main()