
from __future__ import annotations

import os
import time
import tracemalloc
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np

@dataclass
class MatchResult:
    """Stores results for a single pipeline on a single pair."""
    pipeline: str
    good_matches: int
    elapsed_ms: float
    peak_kb: float
    decision: str  # "same" or "different"

def ensure_dir(p: str) -> None:
    """Ensure that a directory exists."""
    os.makedirs(p, exist_ok=True)

def preprocess_fingerprint(img: np.ndarray) -> np.ndarray:
    """Convert image to grayscale and apply Otsu's thresholding (inverted)."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img_bin

def load_image(path: str, fingerprint: bool) -> np.ndarray:
    """Load an image in grayscale. Apply fingerprint preprocessing if required."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return preprocess_fingerprint(img) if fingerprint else img

def _lowe_ratio_good(matches, ratio: float) -> List[cv2.DMatch]:
    """Filter matches using Lowe's ratio test."""
    return [m for m, n in matches if m.distance < ratio * n.distance] if len(matches) else []

def match_orb_bf(img1: np.ndarray, img2: np.ndarray, nfeatures: int = 1000,
                 ratio: float = 0.7, draw: bool = True) -> Tuple[int, Optional[np.ndarray]]:
    """Compute ORB features and match using BFMatcher (Hamming)."""
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return 0, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = _lowe_ratio_good(matches, ratio)
    match_img = None
    if draw:
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, good, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return len(good), match_img

def _sift_create(nfeatures: int = 1000):
    """Create SIFT detector if available; raise if not."""
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError(
            "SIFT is not available in your OpenCV build. "
            "Install/upgrade: pip install --upgrade opencv-contrib-python"
        )
    return cv2.SIFT_create(nfeatures=nfeatures)

def match_sift_flann(img1: np.ndarray, img2: np.ndarray, nfeatures: int = 1000,
                     ratio: float = 0.7, flann_checks: int = 50,
                     draw: bool = True) -> Tuple[int, Optional[np.ndarray]]:
    """Compute SIFT features and match using FLANN KD-tree matcher."""
    sift = _sift_create(nfeatures=nfeatures)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return 0, None
    index_params = dict(algorithm=1, trees=5)  # KD-tree for SIFT
    search_params = dict(checks=flann_checks)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = _lowe_ratio_good(matches, ratio)
    match_img = None
    if draw:
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, good, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return len(good), match_img

def run_with_measurements(fn, *args, **kwargs) -> Tuple[float, float, any]:
    """Measure execution time and peak memory for a function call."""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_kb = peak / 1024.0
    return elapsed_ms, peak_kb, result

def evaluate_pair(img1_path: str, img2_path: str, name: str,
                   fingerprint: bool, threshold: int = 20,
                   out_dir: str = "four_image_results") -> List[MatchResult]:
    """Evaluate both pipelines on a pair of images and save visuals."""
    ensure_dir(out_dir)
    vis_dir = os.path.join(out_dir, f"{name}_visuals")
    ensure_dir(vis_dir)
    img1 = load_image(img1_path, fingerprint)
    img2 = load_image(img2_path, fingerprint)
    results: List[MatchResult] = []
    for pipe_name, matcher in [
        ("ORB+BF", match_orb_bf),
        ("SIFT+FLANN", match_sift_flann),
    ]:
        try:
            # Explicitly pass draw=True so the boolean is assigned to the 'draw' keyword
            elapsed_ms, peak_kb, (good, vis) = run_with_measurements(matcher, img1, img2, draw=True)
        except RuntimeError as e:
            # Handle missing SIFT or other runtime errors gracefully
            print(f"[WARNING] {pipe_name} skipped for {name}: {e}")
            results.append(MatchResult(pipe_name, 0, float('nan'), float('nan'), "error"))
            continue
        decision = "same" if good > threshold else "different"
        # Save visualisation if it was generated
        if vis is not None:
            vis_name = f"{name}_{pipe_name.replace('+','').replace(' ','_')}_{'match' if decision == 'same' else 'nomatch'}_{good}.png"
            cv2.imwrite(os.path.join(vis_dir, vis_name), vis)
        results.append(MatchResult(pipe_name, good, elapsed_ms, peak_kb, decision))
    return results

def main() -> None:
    """Run evaluations on the two pairs defined below and report results."""
    # Update these paths to your own images if needed
    pairs_info = [
        {
            "img1": "101_6.tif",
            "img2": "105_6.tif",
            "name": "fingerprints",
            "fingerprint": True,
        },
        {
            "img1": "UiA front1.png",
            "img2": "UiA front3.jpg",
            "name": "uia",
            "fingerprint": False,
        },
    ]
    # Adjust output folder if desired
    output_base = "four_image_results"
    ensure_dir(output_base)
    # Determine the directory where this script resides so relative image paths work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for info in pairs_info:
        # Construct full paths relative to the script directory
        img1_path = os.path.join(script_dir, info["img1"])
        img2_path = os.path.join(script_dir, info["img2"])
        # Warn if either file is missing
        if not os.path.isfile(img1_path) or not os.path.isfile(img2_path):
            print(f"Image files {img1_path} or {img2_path} not found. Please check the paths.")
            continue
        print(f"\n=== Evaluating {info['name']} ===")
        results = evaluate_pair(img1_path, img2_path, info["name"], info["fingerprint"],
                                threshold=20, out_dir=output_base)
        for res in results:
            print(f"{res.pipeline}: good_matches={res.good_matches}, "
                  f"time={res.elapsed_ms:.1f} ms, peak_mem={res.peak_kb:.1f} KB, decision={res.decision}")

if __name__ == "__main__":
    main()

