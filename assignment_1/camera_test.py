import cv2
import numpy as np
import os
import warnings
import sys

# Suppress the AVCaptureDeviceTypeExternal warning
warnings.filterwarnings("ignore")

def test_camera():
    """Test camera access with different methods"""
    print("Testing camera access...")
    
    # Method 1: Try with different backends
    backends = [
        cv2.CAP_AVFOUNDATION,  # macOS native
        cv2.CAP_ANY,           # Auto-detect
        cv2.CAP_V4L2,          # Linux
        cv2.CAP_DSHOW,         # Windows
    ]
    
    for backend in backends:
        print(f"\nTrying backend: {backend}")
        cap = cv2.VideoCapture(0, backend)
        
        if cap.isOpened():
            # Try to read a frame
            ret, frame = cap.read()
            if ret and frame is not None:
                fps = cap.get(cv2.CAP_PROP_FPS)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                
                print(f"SUCCESS! Camera opened with backend {backend}")
                print(f"fps: {fps}")
                print(f"height: {height}")
                print(f"width: {width}")
                
                # If FPS is 0, calculate manually
                if fps == 0:
                    import time
                    print("FPS reported as 0, calculating manually...")
                    num_frames = 30
                    start = time.time()
                    for i in range(num_frames):
                        ret, frame = cap.read()
                        if not ret:
                            break
                    end = time.time()
                    if end > start:
                        fps = num_frames / (end - start)
                        print(f"Calculated fps: {fps:.1f}")
                
                cap.release()
                
                # Save to file
                os.makedirs('solutions', exist_ok=True)
                with open('solutions/camera_outputs.txt', 'w') as f:
                    f.write(f"fps: {fps:.1f}\n")
                    f.write(f"height: {height}\n")
                    f.write(f"width: {width}\n")
                
                print(f"\nCamera outputs saved to solutions/camera_outputs.txt")
                return True
            
            cap.release()
    
    print("\nCould not access camera with any backend")
    return False

if __name__ == "__main__":
    # Redirect stderr to suppress warnings
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    success = test_camera()
    
    # Restore stderr
    sys.stderr = old_stderr
    
    if not success:
        print("\nPlease check:")
        print("1. Camera permissions in System Settings > Privacy & Security > Camera")
        print("2. That no other application is using the camera")
        print("3. Try running: python camera_test.py")