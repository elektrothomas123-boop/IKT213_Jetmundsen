import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse

# ---------- I. HARRIS CORNER DETECTION ----------
def harris_corners(reference_image_path: str, out_path: str = "harris.png"):
    img_color = cv2.imread(reference_image_path)
    if img_color is None:
        raise FileNotFoundError(f"Could not read {reference_image_path}")

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_gray = np.float32(img_gray)

    # Harris
    dst = cv2.cornerHarris(img_gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)

    # Marker hjørner i rødt
    img_marked = img_color.copy()
    thresh = 0.01 * dst.max()
    img_marked[dst > thresh] = [0, 0, 255]  # BGR

    cv2.imwrite(out_path, img_marked)
    return out_path

# ---------- II. FEATURE-BASED ALIGNMENT ----------
def _sift_detector(max_features=10):
    try:
        sift = cv2.SIFT_create(nfeatures=max_features)
        return sift, True
    except Exception:
        return None, False

def _orb_detector(max_features=1500):
    return cv2.ORB_create(nfeatures=max_features)

def align_images(
    image_to_align_path: str,
    reference_image_path: str,
    method: str = "SIFT",
    max_features: int = 10,
    good_match_percent: float = 0.7,
    out_aligned: str = "aligned.png",
    out_matches: str = "matches.png",
):
    # Les bilder
    im1_color = cv2.imread(image_to_align_path)
    im2_color = cv2.imread(reference_image_path)
    if im1_color is None:
        raise FileNotFoundError(f"Could not read {image_to_align_path}")
    if im2_color is None:
        raise FileNotFoundError(f"Could not read {reference_image_path}")

    im1_gray = cv2.cvtColor(im1_color, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2_color, cv2.COLOR_BGR2GRAY)

    # Velg metode
    use_sift = method.upper() == "SIFT"
    if use_sift:
        detector, ok = _sift_detector(max_features=max_features)
        if not ok:
            print("[WARN] SIFT ikke tilgjengelig – faller tilbake til ORB.")
            use_sift = False

    if not use_sift:
        detector = _orb_detector(max_features=max_features)

    # Keypoints + descriptors
    kp1, des1 = detector.detectAndCompute(im1_gray, None)
    kp2, des2 = detector.detectAndCompute(im2_gray, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        raise RuntimeError("Fikk ikke nok nøkkelpunkter/descriptors til å fortsette.")

    # Matcher
    if use_sift:
        # SIFT: BFMatcher + Lowe ratio
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        raw_matches = bf.knnMatch(des1, des2, k=2)
        good = []
        ratio = good_match_percent  # 0.7 for SIFT
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < ratio * n.distance:
                    good.append(m)
        matches = good
    else:
        # ORB: Hamming + sortering (fiks fra oppgaven)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance, reverse=False)
        num_good = max(4, int(len(matches) * good_match_percent))  # 0.15 for ORB
        matches = matches[:num_good]

    if len(matches) < 4:
        raise RuntimeError(f"For få gode matcher ({len(matches)}) til å beregne homografi.")

    # Punktpar
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Homografi
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("Klarte ikke å beregne homografi (H==None).")

    # Warp
    h, w = im2_color.shape[:2]
    im1_reg = cv2.warpPerspective(im1_color, H, (w, h))

    # Visualisering av matcher
    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=mask.ravel().tolist() if mask is not None else None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    vis_matches = cv2.drawMatches(im1_color, kp1, im2_color, kp2, matches, None, **draw_params)

    cv2.imwrite(out_aligned, im1_reg)
    cv2.imwrite(out_matches, vis_matches)
    return out_aligned, out_matches

# ---------- III. LAG PDF ----------
def images_to_pdf(image_paths, out_pdf="output.pdf"):
    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    if not imgs:
        raise RuntimeError("Ingen bilder å skrive til PDF.")
    first, rest = imgs[0], imgs[1:]
    first.save(out_pdf, save_all=True, append_images=rest)
    return out_pdf

def run_all(method: str):
    # Del I: Harris
    p1 = harris_corners("reference_img.png", out_path="harris.png")

    # Del II: Alignment
    if method.upper() == "SIFT":
        aligned, matches = align_images(
            image_to_align_path="align_this.jpg",
            reference_image_path="reference_img.png",
            method="SIFT",
            max_features=10,
            good_match_percent=0.7,
            out_aligned="aligned.png",
            out_matches="matches.png",
        )
    else:
        aligned, matches = align_images(
            image_to_align_path="align_this.jpg",
            reference_image_path="reference_img.png",
            method="ORB",
            max_features=1500,
            good_match_percent=0.15,
            out_aligned="aligned.png",
            out_matches="matches.png",
        )

    # Del III: PDF
    pdf_path = images_to_pdf(["harris.png", "aligned.png", "matches.png"], out_pdf="output.pdf")
    print("Ferdig! Skrev PDF til:", pdf_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IKT213 Assignment 4")
    parser.add_argument("--method", choices=["SIFT", "ORB"], default="SIFT",
                        help="Velg funksjonsdetektor for alignment (SIFT eller ORB). Default=SIFT.")
    args = parser.parse_args()
    run_all(args.method)
