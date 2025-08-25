import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np


# ---------- IV: Print bilde-informasjon ----------
def print_image_information(image: np.ndarray) -> None:
    if image is None:
        raise ValueError("Bildet er None. Sjekk at 'lena.png' ligger i samme mappe som main.py.")
    shape = image.shape
    if len(shape) == 2:
        h, w = shape
        c = 1
    elif len(shape) == 3:
        h, w, c = shape
    else:
        raise ValueError(f"Uventet bildeshape: {shape}")

    print("=== Image information ===")
    print(f"height: {h}")
    print(f"width: {w}")
    print(f"channels: {c}")
    print(f"size (num values): {image.size}")
    print(f"data type: {image.dtype}")


# ---------- V: Kamera → tekstfil ----------
def _estimate_fps(cap: cv2.VideoCapture, seconds: float = 2.0) -> float:
    """Mål FPS ved å telle frames hvis CAP_PROP_FPS er 0."""
    frames = 0
    start = time.time()
    while time.time() - start < seconds:
        ok, _ = cap.read()
        if not ok:
            break
        frames += 1
    elapsed = time.time() - start
    return frames / elapsed if elapsed > 0 else 0.0


def _try_open_camera(index: int, backends: Iterable[int]) -> Optional[cv2.VideoCapture]:
    """Prøv flere OpenCV-backends og returner en åpen capture eller None."""
    for be in backends:
        cap = cv2.VideoCapture(index, be)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def save_camera_info_txt(output_path: Path, camera_index: int = 0) -> None:
    """
    Skriver fps, height, width til output_path.

    macOS: prøver AVFoundation først (krav for mange maskiner),
    deretter CAP_ANY. Tester også index 1 og 2 hvis 0 feiler.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Velg passende backend-rekkefølge etter OS
    is_macos = sys.platform == "darwin"
    backends = []
    if is_macos and hasattr(cv2, "CAP_AVFOUNDATION"):
        backends.append(cv2.CAP_AVFOUNDATION)
    backends.append(getattr(cv2, "CAP_ANY", 0))

    # Prøv ønsket index, deretter 1 og 2 som fallback
    tried_indices = [camera_index] + [i for i in (1, 2) if i != camera_index]
    cap = None
    used_index = None
    for idx in tried_indices:
        cap = _try_open_camera(idx, backends)
        if cap is not None:
            used_index = idx
            break

    if cap is None:
        raise RuntimeError(
            "Kunne ikke åpne noe webkamera.\n"
            "Sjekk dette:\n"
            " • System Settings → Privacy & Security → Camera → slå på for PyCharm (og ev. Terminal).\n"
            " • Lukk apper som bruker kamera (Teams/Zoom/FaceTime/OBS).\n"
            " • Prøv annet camera_index (0/1/2) i main()."
        )

    # Hent egenskaper
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Estimer FPS hvis rapportert 0
    if not fps or fps <= 1e-3:
        fps = _estimate_fps(cap, seconds=2.0)

    cap.release()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"fps: {fps:.2f}\n")
        f.write(f"height: {height}\n")
        f.write(f"width: {width}\n")

    print(f"🟢 V. Web camera information saved into: {output_path.resolve()} (camera_index={used_index})")


def main():
    # ---------- IV ----------
    img_path = Path(__file__).parent / "lena.png"
    image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    print_image_information(image)

    # ---------- V ----------
    # Lagre i riktig mappe iht. oppgaven
    solutions_dir = Path.home() / "IKT213_lastname" / "assignment_1" / "solutions"
    txt_path = solutions_dir / "camera_outputs.txt"

    # Bytt camera_index ved behov (0 på de fleste, prøv 1/2 hvis flere kamera)
    save_camera_info_txt(txt_path, camera_index=1)


if __name__ == "__main__":
    main()