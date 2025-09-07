import os
import cv2
import numpy as np

# ---------------------------
# Utility
# ---------------------------
def _ensure_outdir():
    outdir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(outdir, exist_ok=True)
    return outdir

def _save(name, img):
    outdir = _ensure_outdir()
    path = os.path.join(outdir, name)
    cv2.imwrite(path, img)
    print(f"Saved: {path}")

# ---------------------------
# I. Functions (exact names & args)
# ---------------------------

def padding(image, border_width):
    """
    Create a border around the image by reflecting edges.
    """
    return cv2.copyMakeBorder(
        image,
        top=border_width,
        bottom=border_width,
        left=border_width,
        right=border_width,
        borderType=cv2.BORDER_REFLECT
    )

def crop(image, x_0, x_1, y_0, y_1):
    """
    Return the sub-image image[y_0:y_1, x_0:x_1].
    NOTE: x = columns (width), y = rows (height).
    """
    h, w = image.shape[:2]
    # Clamp to image bounds for safety
    x_0 = max(0, min(x_0, w))
    x_1 = max(0, min(x_1, w))
    y_0 = max(0, min(y_0, h))
    y_1 = max(0, min(y_1, h))
    if x_0 >= x_1 or y_0 >= y_1:
        raise ValueError("Invalid crop coordinates after clamping.")
    return image[y_0:y_1, x_0:x_1]

def resize(image, width, height):
    """
    Resize to (width, height). Uses INTER_AREA (good for downscaling).
    """
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def copy(image, emptyPictureArray):
    """
    Manual pixel copy into the provided empty array.
    Do NOT use cv2.copy(); use numpy assignment only.
    """
    if emptyPictureArray.shape != image.shape:
        raise ValueError("emptyPictureArray must have the same shape as image.")
    # Vectorized copy, no OpenCV copy
    emptyPictureArray[:] = image
    return emptyPictureArray

def grayscale(image):
    """
    Convert BGR image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def hsv(image):
    """
    Convert BGR image to HSV (OpenCV uses H in [0,179]).
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def hue_shifted(image, emptyPictureArray, hue):
    """
    Shift color values of the (B,G,R) image by 'hue' for ALL channels,
    using wrap-around within [0,255] as hinted in the assignment.
    (Name kept as requested.)
    """
    if emptyPictureArray.shape != image.shape:
        raise ValueError("emptyPictureArray must have the same shape as image.")
    # Wrap-around addition on uint8:
    shifted = (image.astype(np.int16) + int(hue)) % 256
    emptyPictureArray[:] = shifted.astype(np.uint8)
    return emptyPictureArray

def smoothing(image):
    """
    Gaussian blur with ksize=(15,15) and default border.
    """
    return cv2.GaussianBlur(image, (15, 15), 0, borderType=cv2.BORDER_DEFAULT)

def rotation(image, rotation_angle):
    """
    Rotate image by exactly 90° clockwise or 180°, per spec.
    """
    if rotation_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    else:
        raise ValueError("rotation_angle must be 90 or 180.")

# ---------------------------
# II. Run the required tasks
# ---------------------------
if __name__ == "__main__":
    here = os.path.dirname(__file__)
    img_path = os.path.join(here, "lena.png")
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(
            "Could not read lena.png. Place it in assignment_2/ next to main.py."
        )

    h, w = img.shape[:2]

    # Padding: reflect, border_width=100
    padded = padding(img, border_width=100)
    _save("lena_padded_reflect_100.png", padded)

    # Cropping: 80 px from left/top; 130 px from right/bottom
    x0, y0 = 80, 80
    x1, y1 = w - 130, h - 130
    cropped = crop(img, x0, x1, y0, y1)
    _save("lena_cropped_80_130.png", cropped)

    # Resize: 200x200
    resized = resize(img, 200, 200)
    _save("lena_resized_200x200.png", resized)

    # Manual copy (no cv2.copy())
    empty = np.zeros_like(img, dtype=np.uint8)
    copied = copy(img, empty)
    _save("lena_copied_manual.png", copied)

    # Grayscale
    gray = grayscale(img)
    _save("lena_grayscale.png", gray)

    # HSV
    hsv_img = hsv(img)
    _save("lena_hsv.png", hsv_img)  # Note: HSV saved directly may look odd in viewers—this is expected.

    # Color shifting: shift by +50 for all color values (wrap-around)
    empty2 = np.zeros_like(img, dtype=np.uint8)
    shifted = hue_shifted(img, empty2, hue=50)
    _save("lena_color_shift_plus50.png", shifted)

    # Smoothing: Gaussian blur (15x15)
    smooth = smoothing(img)
    _save("lena_blur_15x15.png", smooth)

    # Rotation: 90° clockwise and 180°
    rot90 = rotation(img, 90)
    _save("lena_rot_90_clockwise.png", rot90)

    rot180 = rotation(img, 180)
    _save("lena_rot_180.png", rot180)

    print("All tasks completed.")