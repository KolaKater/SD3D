__VERSION__ = "V75.31 - Anti-Alias + Output Type separated + GFPGAN V75.25 + Text Fix V75.30"
import argparse, gc, os, pathlib, random, re
import numpy as np
import torch
from os.path import basename, splitext, join
from typing import Optional, List
from PIL import Image
import cv2
DEVICE = "cuda"

def _is_surface_blur_number(s):
    try:
        return float(s) > 0
    except:
        return False
def _get_surface_blur_float(s):
    try:
        return float(s)
    except:
        return 0.0

try:
    import torchvision.transforms.functional as TF_F
    TF = TF_F
except:
    TF_F = None
    TF = None
from GFPGAN_face import init_face_restorer, restore_face
from text_fix import preserve_text_regions, preserve_line_regions
from output_type import create_sbs_image, OUTPUT_TYPE_OPTIONS as OT_OPTIONS

# V75.31 Anti-Alias separated to anti_alias.py
try:
    from anti_alias import apply_antialias, smooth_antialias_stretch, apply_surface_blur, apply_antialias_to_sbs
except ImportError:
    # Fallback local implementation if anti_alias.py missing
    def apply_surface_blur(pil_img, radius=0):
        if radius <=0:
            return pil_img
        try:
            img_np = np.array(pil_img)
            blurred = cv2.bilateralFilter(img_np, d=int(radius*2+1), sigmaColor=75, sigmaSpace=75)
            return Image.fromarray(blurred)
        except:
            return pil_img
    def apply_antialias(pil_img, amount=0):
        if abs(amount) < 0.01:
            return pil_img
        try:
            img_np = np.array(pil_img)
            if amount > 0.05:
                blurred = cv2.GaussianBlur(img_np, (0,0), sigmaX=amount, sigmaY=amount)
                return Image.fromarray(blurred)
            elif amount < -0.05:
                amt = abs(amount)
                blurred = cv2.GaussianBlur(img_np, (0,0), 1.0, 1.0)
                sharpened = cv2.addWeighted(img_np, 1.0 + amt, blurred, -amt, 0)
                sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
                return Image.fromarray(sharpened)
            else:
                return pil_img
        except:
            return pil_img
    def smooth_antialias_stretch(pil_img, target_w, target_h, antialias=0, is_original=False):
        img_np = np.array(pil_img)
        if antialias > 0.05:
            img_np = cv2.GaussianBlur(img_np, (0,0), sigmaX=antialias, sigmaY=antialias)
        elif antialias < -0.05:
            amount = abs(antialias)
            blurred = cv2.GaussianBlur(img_np, (0,0), 1.0, 1.0)
            img_np = cv2.addWeighted(img_np, 1.0 + amount, blurred, -amount, 0)
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        resized = cv2.resize(img_np, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        return Image.fromarray(resized)
    def apply_antialias_to_sbs(sbs_image, amount=0, log_func=None):
        if abs(amount) < 0.01:
            return sbs_image
        try:
            sbs_np = np.array(sbs_image)
            if amount > 0.05:
                sbs_np = cv2.GaussianBlur(sbs_np, (0,0), sigmaX=amount, sigmaY=amount)
            elif amount < -0.05:
                amt = abs(amount)
                blurred = cv2.GaussianBlur(sbs_np, (0,0), 1.0, 1.0)
                sbs_np = cv2.addWeighted(sbs_np, 1.0 + amt, blurred, -amt, 0)
                sbs_np = np.clip(sbs_np, 0, 255).astype(np.uint8)
            return Image.fromarray(sbs_np)
        except:
            return sbs_image
def get_autocast():
    try:
        return torch.amp.autocast("cuda")
    except:
        try:
            from torch.cuda.amp import autocast
            return autocast()
        except:
            from contextlib import nullcontext
            return nullcontext()
def get_aspect_restored_size(orig_w, orig_h, base=768):
    if orig_w >= orig_h:
        new_w = base * orig_w // orig_h if orig_h else base
        new_h = base
    else:
        new_h = base * orig_h // orig_w if orig_w else base
        new_w = base
    return new_w, new_h
# smooth_antialias_stretch now in anti_alias.py - removed
def apply_gamma(pil_img, gamma=1.0):
    if abs(gamma - 1.0) < 0.001 or pil_img is None:
        return pil_img
    try:
                    # global np used
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        corrected = np.power(img_np, 1.0 / gamma)
                    # global Image used
        return Image.fromarray((np.clip(corrected * 255.0, 0, 255)).astype(np.uint8))
    except:
        return pil_img
def apply_gamma_correction(pil_img, gamma=1.0):
    return apply_gamma(pil_img, gamma)
def process_image(src_path, src_intrinsics=None, tgt_intrinsics=None, crop_size=768, resize_mode="stretch", gamma_a=1.0):
    from os.path import basename, splitext
    import torch
    import torchvision.transforms as transforms
    import torchvision.transforms.functional as TF_F
                    # global Image used
    src_img = Image.open(src_path).convert("RGB")
    if abs(gamma_a - 1.0) > 0.001:
        try:
                    # global np used
            img_np = np.array(src_img).astype(np.float32) / 255.0
            corrected = np.power(img_np, 1.0 / gamma_a)
            src_img = Image.fromarray((np.clip(corrected * 255.0, 0, 255)).astype(np.uint8))
            print(f"[Gamma A] V84 FIXED Applied {gamma_a} to ORIGINAL {basename(src_path)} before {crop_size}x{crop_size} resize", flush=True)
        except Exception as e:
            print(f"[Gamma A] V84 Failed {e}")
    w, h = src_img.size
    if resize_mode == "stretch":
        src_resized = src_img.resize((crop_size, crop_size), Image.Resampling.LANCZOS)
        canvas = src_resized
        left = 0; top = 0; new_w = crop_size; new_h = crop_size
        scale_w = crop_size / w; scale_h = crop_size / h; scale = (scale_w + scale_h)/2
    else:
        scale = crop_size / max(w, h)
        new_w = int(round(w * scale)); new_h = int(round(h * scale))
        src_resized = src_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (crop_size, crop_size), (0, 0, 0))
        left = (crop_size - new_w)//2; top = (crop_size - new_h)//2
        canvas.paste(src_resized, (left, top))
    if src_intrinsics is not None:
        src_K = src_intrinsics.clone().float()
        if resize_mode == "stretch":
            src_K[0,0] *= scale_w; src_K[1,1] *= scale_h
            src_K[0,2] *= scale_w; src_K[1,2] *= scale_h
        else:
            src_K[0,0] *= scale; src_K[1,1] *= scale
            src_K[0,2] *= scale; src_K[1,2] *= scale
            src_K[0,2] += left; src_K[1,2] += top
        src_K = src_K.unsqueeze(0)
    else:
        src_K = None
    if tgt_intrinsics is not None:
        tgt_K = tgt_intrinsics.clone().float()
        if resize_mode == "stretch":
            tgt_K[0,0] *= scale_w; tgt_K[1,1] *= scale_h
            tgt_K[0,2] *= scale_w; tgt_K[1,2] *= scale_h
        else:
            tgt_K[0,0] *= scale; tgt_K[1,1] *= scale
            tgt_K[0,2] *= scale; tgt_K[1,2] *= scale
            tgt_K[0,2] += left; tgt_K[1,2] += top
        tgt_K = tgt_K.unsqueeze(0)
    elif src_K is not None:
        tgt_K = src_K.clone()
    else:
        tgt_K = None
    src_crop = canvas
    transform_pixels = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_tensor = transform_pixels(src_crop).to(dtype=dtype, device=device)
    src = src_tensor.unsqueeze(0)
    base_name = splitext(basename(src_path))[0]
    letterbox = (left, top, new_w, new_h)
    orig_w, orig_h = w, h
    orig_pil = src_img.copy()
    return src, src_K, tgt_K, base_name, letterbox, orig_pil, (orig_w, orig_h)
def collect_image_paths(input_path):
    p = pathlib.Path(input_path)
    if p.is_file():
        return [str(p)]
    exts = {".jpg",".jpeg",".png",".bmp",".webp"}
    paths = []
    for f in p.iterdir():
        if f.suffix.lower() in exts:
            paths.append(str(f))
    return sorted(paths)
def stack_optional(tensors):
    if tensors[0] is None:
        return None
    return torch.stack([torch.from_numpy(t) if isinstance(t, np.ndarray) else t for t in tensors])
def parse_enhance(enhance_str):
    if not enhance_str or enhance_str.lower() == "none":
        return None, None, None
    s = enhance_str.lower()
    model_type = "realesrgan"
    target = None
    key = ""
    if "upscayl" in s:
        model_type = "upscayl"
    elif "clarity" in s:
        model_type = "clarity"
    elif "maxine" in s:
        model_type = "maxine"
    elif "realesrgan" in s or "esrgan" in s:
        model_type = "realesrgan"
    elif "rtx_denoise" in s or "denoise" in s:
        model_type = "rtx_denoise"
    elif "rtx" in s or "nis" in s:
        model_type = "rtx"
    if "fhd" in s or "1920" in s:
        target = 1920
        key = "fhd"
    elif "2k" in s or "2560" in s:
        target = 2560
        key = "2k"
    elif "4k" in s or "3840" in s:
        target = 3840
        key = "4k"
    else:
        import re
        m = re.search(r"(\d+)k", s)
        if m:
            target = int(m.group(1))*1000
            key = f"{m.group(1)}k"
        else:
            target = 1920
            key = "fhd"
    print(f"[ENHANCE] Mode: {enhance_str} -> model {model_type}, target {target}, key {key} (V84 FIX)")
    return model_type, target, key


def _forbidden_zones(orig_pil):
    """Forbidden x-ranges: face + large objects + skin blobs. Seam must NOT fall inside."""
    import numpy as _np
    import cv2 as _cv2
    W, H = orig_pil.size
    img = _np.asarray(orig_pil).astype(_np.uint8)
    gray = _cv2.cvtColor(img, _cv2.COLOR_RGB2GRAY)
    zones = []
    # Face: Haar if available, else skin-blob fallback (headless cv2 has no CascadeClassifier)
    _haar_ok = False
    try:
        if hasattr(_cv2, "CascadeClassifier"):
            cascade = _cv2.CascadeClassifier(_cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
            for (x, y, w, h) in faces:
                zones.append((max(0, int(x - 40)), min(W, int(x + w + 40)), "face"))
            _haar_ok = len(faces) > 0
            if _haar_ok:
                pass  # [SD3D] log suppressed (Easy3D-style)
    except Exception as e:
        pass  # [SD3D] log suppressed (Easy3D-style)
    if not _haar_ok:
        try:
            # Skin-blob face proxy: skin mask, close, contours with face-like size/aspect
            ycrcb_f = _cv2.cvtColor(img, _cv2.COLOR_RGB2YCrCb)
            mask_f = _cv2.inRange(ycrcb_f, _np.array([0, 135, 85]), _np.array([255, 180, 135]))
            mask_f = _cv2.morphologyEx(mask_f, _cv2.MORPH_CLOSE, _np.ones((15, 15), _np.uint8), iterations=1)
            cnts_f, _ = _cv2.findContours(mask_f, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
            _n = 0
            for c in cnts_f:
                area = _cv2.contourArea(c)
                if area < 0.01 * W * H or area > 0.15 * W * H:
                    continue
                x, y, w, h = _cv2.boundingRect(c)
                if h < 0.15 * H:
                    continue
                aspect = w / max(1, h)
                if aspect < 0.5 or aspect > 1.8:
                    continue
                zones.append((max(0, int(x - 30)), min(W, int(x + w + 30)), "face-skin"))
                _n += 1
            if _n:
                pass  # [SD3D] log suppressed (Easy3D-style)
        except Exception as e2:
            pass  # [SD3D] log suppressed (Easy3D-style)
    try:
        edges = _cv2.Canny(gray, 50, 150)
        dil = _cv2.dilate(edges, _np.ones((9, 9), _np.uint8), iterations=1)
        cnts, _ = _cv2.findContours(dil, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if _cv2.contourArea(c) < 0.04 * W * H:
                continue
            x, y, w, h = _cv2.boundingRect(c)
            if h < 0.15 * H:
                continue
            if w > 0.60 * W:
                continue  # FIX: wider than 60% = background, not an object
            if _cv2.contourArea(c) > 0.50 * W * H:
                continue
            zones.append((max(0, int(x - 24)), min(W, int(x + w + 24)), "large"))
    except Exception as e:
        pass  # [SD3D] log suppressed (Easy3D-style)
    try:
        ycrcb = _cv2.cvtColor(img, _cv2.COLOR_RGB2YCrCb)
        mask = _cv2.inRange(ycrcb, _np.array([0, 135, 85]), _np.array([255, 180, 135]))
        mask = _cv2.morphologyEx(mask, _cv2.MORPH_CLOSE, _np.ones((15, 15), _np.uint8), iterations=1)
        cnts2, _ = _cv2.findContours(mask, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts2:
            if _cv2.contourArea(c) < 0.01 * W * H:
                continue
            x, y, w, h = _cv2.boundingRect(c)
            if h < 0.10 * H:
                continue
            if w > 0.60 * W:
                continue  # FIX: wider than 60% = warm background, not skin blob
            zones.append((max(0, int(x - 20)), min(W, int(x + w + 20)), "skin"))
    except Exception as e:
        pass  # [SD3D] log suppressed (Easy3D-style)
    return zones


def _generic_object_zones(orig_pil):
    """Generic object zones: circles (basin/bucket/plate/wheel) + large contours (door/table/box/body).
    No Haar, no skin - works on any image, any colour temperature."""
    import numpy as _np
    import cv2 as _cv2
    W, H = orig_pil.size
    img = _np.asarray(orig_pil).astype(_np.uint8)
    gray = _cv2.cvtColor(img, _cv2.COLOR_RGB2GRAY)
    zones = []
    # --- 1) Circles: basin / bucket / plate / wheel / clock / reel ---
    try:
        blur = _cv2.medianBlur(gray, 5)
        circles = _cv2.HoughCircles(blur, _cv2.HOUGH_GRADIENT, dp=1.2, minDist=80,
                                     param1=100, param2=80, minRadius=30, maxRadius=220)
        if circles is not None:
            circles = _np.uint16(_np.around(circles))
            for (cx, cy, r) in circles[0, :]:
                zones.append((max(0, int(cx - r - 20)), min(W, int(cx + r + 20)), f"circle r={int(r)}"))
    except Exception as e:
        pass  # [SD3D] log suppressed (Easy3D-style)
    # --- 2) Large contours: door / table / box / body / machine ---
    try:
        edges = _cv2.Canny(gray, 30, 100)
        dil = _cv2.dilate(edges, _np.ones((9, 9), _np.uint8), iterations=1)
        cnts, _ = _cv2.findContours(dil, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = _cv2.contourArea(c)
            if area < 0.02 * W * H or area > 0.50 * W * H:
                continue
            x, y, w, h = _cv2.boundingRect(c)
            if h < 0.10 * H:
                continue
            if w > 0.80 * W:
                continue  # wider than 80% = background, not an object
            zones.append((max(0, int(x - 20)), min(W, int(x + w + 20)), "large"))
    except Exception as e:
        pass  # [SD3D] log suppressed (Easy3D-style)
    return zones


def _forbidden_zones_xy(orig_pil):
    """Like _forbidden_zones but with y-range: (x0,x1,y0,y1,kind). For wavy seam per-row penalty.
    Face box (x,y,w,h) -> (x-40, x+w+40, y-20, y+h+20). Circle (cx,cy,r) -> (cx-r-20, cx+r+20, cy-r-20, cy+r+20)."""
    import numpy as _np
    import cv2 as _cv2
    W, H = orig_pil.size
    img = _np.asarray(orig_pil).astype(_np.uint8)
    gray = _cv2.cvtColor(img, _cv2.COLOR_RGB2GRAY)
    zones = []
    try:
        if hasattr(_cv2, "CascadeClassifier"):
            cascade = _cv2.CascadeClassifier(_cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
            for (x, y, w, h) in faces:
                zones.append((max(0, int(x-40)), min(W, int(x+w+40)), max(0, int(y-20)), min(H, int(y+h+20)), "face"))
    except Exception: pass
    try:
        blur = _cv2.medianBlur(gray, 5)
        circles = _cv2.HoughCircles(blur, _cv2.HOUGH_GRADIENT, dp=1.2, minDist=80, param1=100, param2=80, minRadius=30, maxRadius=220)
        if circles is not None:
            circles = _np.uint16(_np.around(circles))
            for (cx, cy, r) in circles[0, :]:
                zones.append((max(0, int(cx-r-20)), min(W, int(cx+r+20)), max(0, int(cy-r-20)), min(H, int(cy+r+20)), f"circle r={int(r)}"))
    except Exception: pass
    try:
        edges = _cv2.Canny(gray, 30, 100)
        dil = _cv2.dilate(edges, _np.ones((9, 9), _np.uint8), iterations=1)
        cnts, _ = _cv2.findContours(dil, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = _cv2.contourArea(c)
            if area < 0.02 * W * H or area > 0.50 * W * H:
                continue
            x, y, w, h = _cv2.boundingRect(c)
            if h < 0.10 * H or w > 0.80 * W:
                continue
            zones.append((max(0, int(x-20)), min(W, int(x+w+20)), max(0, int(y-10)), min(H, int(y+h+10)), "large"))
    except Exception: pass
    return zones

def _find_wavy_seam(orig_pil, ox0, ox1, margin=32, smooth=2.0):
    """Wavy seam: one x per row, DP. Seam snakes around face/circle/door, not one straight line.
    Returns array shape (H,) with x in [ox0+margin, ox1-margin] per row."""
    try:
        import numpy as _np
        import cv2 as _cv2
        H, W = orig_pil.size[1], orig_pil.size[0]
        gray = _cv2.cvtColor(_np.asarray(orig_pil).astype(_np.uint8), _cv2.COLOR_RGB2GRAY).astype(_np.float64)
        gx = _np.abs(_np.diff(gray, axis=1))
        gy = _np.abs(_np.diff(gray, axis=0))
        grad = _np.zeros_like(gray)
        grad[:, :-1] += gx; grad[:-1, :] += gy
        try:
            grad = _cv2.GaussianBlur(grad.astype(_np.float32), (0, 0), 10.0).astype(_np.float64)
        except: pass
        try:
            _gx = _np.abs(_cv2.Sobel(_np.asarray(orig_pil).astype(_np.uint8), _cv2.CV_64F, 1, 0, ksize=3))
            _gx = _cv2.GaussianBlur(_gx.astype(_np.float32), (0, 0), 10.0).astype(_np.float64)
            grad = grad + 1.0 * _gx
        except: pass
        # Forbidden PER-ROW: face/circle/large only ban the rows they actually occupy.
        # Straight version bans 698-883 for ALL rows, so wavy could only move 608..633. Per-row lets it snake.
        _zone_xy = []
        try:
            _zone_xy = _forbidden_zones_xy(orig_pil)
            _zone_xy = [z for z in _zone_xy if z[1] > ox0 and z[0] < ox1]
        except Exception as e:
            pass  # [SD3D] log suppressed (Easy3D-style)
        if _zone_xy:
            pass  # [SD3D] log suppressed (Easy3D-style)
        _zone_list = _zone_xy  # for later print compat
        lo = ox0 + margin
        hi = ox1 - margin
        if hi - lo < 8:
            lo, hi = ox0, ox1
        n = hi - lo
        # Cost: H x n, grad + forbidden penalty (1e6 inside zone)
        cost = grad[:, lo:hi].copy()
        _xs = _np.arange(lo, hi, dtype=_np.float64)
        # Per-row penalty: 1e6 only if (y in [y0,y1] AND x in [x0,x1])
        _pen2d = _np.zeros_like(cost)
        for (_z0, _z1, _zy0, _zy1, _k) in _zone_xy:
            _xmask = ((_xs >= _z0) & (_xs <= _z1))
            _pen2d[_zy0:_zy1+1, :] += _xmask[None, :] * 1e6
        cost = cost + _pen2d
        # DP: dp[y, x] = cost + min(dp[y-1, x-1..x+1] + smooth*|dx|)
        # Downsample rows for speed: every 4th row, then upsample seam (TH up to 756, n up to 128, H*n ~ 96k, fine at full res)
        Hh = cost.shape[0]
        dp = cost[0].copy()
        back = _np.zeros((Hh, n), dtype=_np.int32)
        for y in range(1, Hh):
            prev = dp
            # 3 candidates: from x-1, x, x+1
            _left = _np.roll(prev, 1); _left[0] = 1e18
            _right = _np.roll(prev, -1); _right[-1] = 1e18
            # Stack: 0=left (dx=+1), 1=center (dx=0), 2=right (dx=-1)
            cand = _np.stack([_left + smooth, prev, _right + smooth], axis=0)  # 3 x n
            best = _np.argmin(cand, axis=0)
            back[y] = best
            dp = cost[y] + cand[best, _np.arange(n)]
        # Backtrack
        seam = _np.zeros(Hh, dtype=_np.int32)
        seam[-1] = int(_np.argmin(dp))
        for y in range(Hh - 1, 0, -1):
            b = back[y, seam[y]]
            if b == 0:
                seam[y - 1] = max(0, seam[y] - 1)
            elif b == 2:
                seam[y - 1] = min(n - 1, seam[y] + 1)
            else:
                seam[y - 1] = seam[y]
        seam_x = (seam + lo).astype(_np.float64)
        # Smooth seam_x over y (median 9) so it does not jitter row-to-row
        try:
            k = 9
            pad = _np.pad(seam_x, (k // 2, k // 2), mode="edge")
            seam_x = _np.array([_np.median(pad[i:i + k]) for i in range(len(seam_x))])
        except: pass
        _dp_min, _dp_max = int(seam_x.min()), int(seam_x.max())
        # SINE WAVY: SBS is left-right, seam is vertical, so wave must oscillate LEFT-RIGHT.
        # User drawing: vertical lines waving left-right, AMP ~28, PERIOD ~96.
        # DP alone only drifts 608..652 (slow), eye still sees a straight vertical line.
        # Sine breaks the straight line: seam_x(y) = base(y) + AMP*sin(2*pi*y/PERIOD)
        WAVY_AMP = 12.0    # <-- knob: 8=very gentle, 12=default (was 28, smeared), 20=strong
        WAVY_PERIOD = 128.0  # <-- knob: 96=fast, 128=default, 192=slow
        try:
            _ys = _np.arange(len(seam_x), dtype=_np.float64)
            _sine = WAVY_AMP * _np.sin(2.0 * _np.pi * _ys / WAVY_PERIOD)
            _cand = seam_x + _sine
            # Clip to overlap [ox0+margin, ox1-margin] - actually lo..hi from above
            _cand = _np.clip(_cand, lo, hi)
            # If sine pushes into a forbidden zone for that row, keep DP base for that row
            for (_z0, _z1, _zy0, _zy1, _k) in _zone_xy:
                _rows = _np.arange(max(0, _zy0), min(len(seam_x), _zy1 + 1))
                if len(_rows) == 0:
                    continue
                _inside = (_cand[_rows] >= _z0) & (_cand[_rows] <= _z1)
                if _np.any(_inside):
                    _cand[_rows[_inside]] = seam_x[_rows[_inside]]
            seam_x = _cand
            pass  # [SD3D] log suppressed (Easy3D-style)
        except Exception as _se:
            pass  # [SD3D] log suppressed (Easy3D-style)
        pass  # [SD3D] log suppressed (Easy3D-style)
        return seam_x
    except Exception as e:
        pass  # [SD3D] log suppressed (Easy3D-style)
        return None

def _find_softest_seam(orig_pil, ox0, ox1, margin=32):
    """Auto: flattest column in overlap [ox0, ox1] from ORIG (not generated tiles).
    Lowest Sobel gradient = wall/sky/flowers, NOT face/hand/table edge. Fixes Sisi top.
    Returns x in [ox0+margin, ox1-margin], or mid if overlap too small."""
    try:
        import numpy as _np
        import cv2 as _cv2
        gray = _cv2.cvtColor(_np.asarray(orig_pil).astype(_np.uint8), _cv2.COLOR_RGB2GRAY).astype(_np.float64)
        gx = _np.abs(_np.diff(gray, axis=1))
        gy = _np.abs(_np.diff(gray, axis=0))
        grad = _np.zeros_like(gray)
        grad[:, :-1] += gx; grad[:-1, :] += gy
        # SOFT SEAM: blur gradient 21x21 before scoring - picks a neighbourhood that is soft,
        # not one lucky column between two flowers. Fixes portrait (flowers+light everywhere).
        try:
            grad = _cv2.GaussianBlur(grad.astype(_np.float32), (0, 0), 10.0).astype(_np.float64)
        except: pass
        # Vertical-edge penalty: seam is vertical, a vertical edge (door frame / wall edge) at seam = visible line.
        # Sobel-X * 1.0 extra, so door frame at 735 is penalised even if total gradient is low.
        try:
            _gx = _np.abs(_cv2.Sobel(_np.asarray(orig_pil).astype(_np.uint8), _cv2.CV_64F, 1, 0, ksize=3))
            _gx = _cv2.GaussianBlur(_gx.astype(_np.float32), (0, 0), 10.0).astype(_np.float64)
            grad = grad + 1.0 * _gx
        except: pass
        # FORBIDDEN: face + large objects + skin - seam must avoid even if flat
        try:
            _zones = _forbidden_zones(orig_pil)
            _zone_list = [(z0, z1, kind) for (z0, z1, kind) in _zones if z1 > ox0 and z0 < ox1]
            # Generic objects: circles (basin/bucket) + large (door/table) - face is just one kind
            try:
                _gz = _generic_object_zones(orig_pil)
                _gz_in = [(z0, z1, kind) for (z0, z1, kind) in _gz if z1 > ox0 and z0 < ox1]
                if _gz_in:
                    pass  # [SD3D] log suppressed (Easy3D-style)
                    _zone_list = _zone_list + _gz_in
            except Exception as _ge:
                pass  # [SD3D] log suppressed (Easy3D-style)
            if _zone_list:
                pass  # [SD3D] log suppressed (Easy3D-style)
        except Exception as _ze:
            _zone_list = []
            pass  # [SD3D] log suppressed (Easy3D-style)
        if ox1 - ox0 > 2 * margin + 8:
            _seg = grad[:, ox0 + margin:ox1 - margin]
            cand = _seg.mean(axis=0) + 0.5 * _seg.std(axis=0)
            _xs1 = _np.arange(ox0 + margin, ox1 - margin, dtype=_np.float64)
            _pen1 = _np.zeros_like(_xs1)
            for (_z0, _z1, _k) in _zone_list:
                _pen1 += ((_xs1 >= _z0) & (_xs1 <= _z1)) * 1e6
            cand = cand + _pen1
            return int(ox0 + margin + int(_np.argmin(cand)))
        _seg = grad[:, ox0:ox1]
        cand = _seg.mean(axis=0) + 0.5 * _seg.std(axis=0)
        return int(ox0 + int(_np.argmin(cand)))
    except Exception as e:
        pass  # [SD3D] log suppressed (Easy3D-style)
        return (ox0 + ox1) // 2


def _set_train_768(stereo_nvs, config=None):
    """Temporarily force train_width/height=768 for tiling. Returns restore fn."""
    old = {}
    for obj, label in [(config, "config"), (getattr(stereo_nvs, "config", None), "stereo_config"), (getattr(stereo_nvs, "cfg", None), "stereo_cfg")]:
        if obj is None: continue
        try:
            # config.data.train_width or config.train_width
            for holder, name in [(getattr(obj, "data", None), "data"), (obj, "top")]:
                if holder is None: continue
                if hasattr(holder, "train_width"):
                    old[(label, name, "w")] = int(getattr(holder, "train_width"))
                    holder.train_width = 768
                if hasattr(holder, "train_height"):
                    old[(label, name, "h")] = int(getattr(holder, "train_height"))
                    holder.train_height = 768
        except Exception: pass
    # Also direct attrs on stereo_nvs
    for attr in ["train_width", "train_height"]:
        try:
            if hasattr(stereo_nvs, attr):
                old[("stereo", attr)] = int(getattr(stereo_nvs, attr))
                setattr(stereo_nvs, attr, 768)
        except Exception: pass
    def _restore():
        for (label, name, wh), val in list(old.items()):
            try:
                if label == "config": obj = config
                elif label == "stereo_config": obj = getattr(stereo_nvs, "config", None)
                elif label == "stereo_cfg": obj = getattr(stereo_nvs, "cfg", None)
                elif label == "stereo": setattr(stereo_nvs, name, val); continue
                else: continue
                if obj is None: continue
                holder = getattr(obj, "data", None) if name == "data" else obj
                if holder is None: continue
                if wh == "w": holder.train_width = val
                else: holder.train_height = val
            except Exception: pass
    return _restore


def _align_tiles_per_row(a, b, x0a, x0b, ox0, ox1, TH):
    # DISABLED: whole-tile warp smeared the image. Return tiles unchanged.
    import numpy as _np_noop
    return a, b, _np_noop.zeros(TH)
    """Per-row disparity align: estimate dx(y) in overlap, warp A +dx/2, B -dx/2 to middle.
    a,b: float64 HxWx3 tiles on full canvas coords (already placed, but here passed as tile arrays).
    Actually a,b are tile arrays (th x tw x 3) with origins x0a,x0b. ox0..ox1 is overlap in full coords.
    Returns (a2, b2, dx_smooth) where a2/b2 are warped tile arrays, dx_smooth is (TH,) array."""
    import numpy as _np
    import cv2 as _cv2
    BAND = 16
    MAX_SHIFT = 24
    # Overlap in tile-local coords
    a_lo, a_hi = ox0 - x0a, ox1 - x0a
    b_lo, b_hi = ox0 - x0b, ox1 - x0b
    if a_lo < 0 or b_lo < 0 or a_hi > a.shape[1] or b_hi > b.shape[1]:
        return a, b, _np.zeros(TH)
    H = a.shape[0]
    # Estimate dx per band centre
    centres = list(range(BAND // 2, H, BAND))
    dxs = []
    for yc in centres:
        y0 = max(0, yc - 16); y1 = min(H, yc + 16)
        if y1 - y0 < 8:
            dxs.append(0.0); continue
        A = a[y0:y1, a_lo:a_hi].astype(_np.float32)
        B = b[y0:y1, b_lo:b_hi].astype(_np.float32)
        # If band is flat (wall), SSD is noisy -> dx=0
        if float(A.std()) < 5.0 and float(B.std()) < 5.0:
            dxs.append(0.0); continue
        best_dx, best_ssd = 0, 1e18
        for dx in range(-MAX_SHIFT, MAX_SHIFT + 1):
            # Shift B by dx to align to A: compare A[:, :] vs B shifted
            # B shifted +dx means B[:, i] comes from B[:, i-dx]
            if dx >= 0:
                # B[dx:] vs A[: -dx]
                if A.shape[1] - dx < 16:
                    continue
                diff = A[:, :A.shape[1] - dx] - B[:, dx:]
            else:
                if A.shape[1] + dx < 16:
                    continue
                diff = A[:, -dx:] - B[:, :B.shape[1] + dx]
            ssd = float((diff * diff).mean())
            if ssd < best_ssd:
                best_ssd, best_dx = ssd, dx
        dxs.append(float(best_dx))
    dxs = _np.array(dxs, dtype=_np.float64)
    # Smooth over y: median 5 + Gaussian
    try:
        # Interpolate to full H
        _xs = _np.array(centres, dtype=_np.float64)
        _ys = _np.arange(H, dtype=_np.float64)
        dx_full = _np.interp(_ys, _xs, dxs)
        # Median 5
        k = 5
        pad = _np.pad(dx_full, (k // 2, k // 2), mode="edge")
        dx_full = _np.array([_np.median(pad[i:i + k]) for i in range(len(dx_full))])
        # Gaussian sigma 8
        dx_full = _cv2.GaussianBlur(dx_full.astype(_np.float32).reshape(-1, 1), (0, 0), 8.0).reshape(-1).astype(_np.float64)
    except Exception:
        dx_full = _np.interp(_np.arange(H, dtype=_np.float64), _np.array(centres, dtype=_np.float64), dxs)
    dx_full = _np.clip(dx_full, -MAX_SHIFT, MAX_SHIFT)
    # Warp: A +dx/2, B -dx/2. Per-row horizontal shift via linear interpolation.
    def _warp_tile(tile, shift_arr, is_a):
        # shift_arr: (H,) shift for this tile (+ means content moves right)
        # out[y, x] = tile[y, x - s] via per-channel np.interp (simple, no shape bug)
        out = tile.copy()
        Hh, Ww = tile.shape[:2]
        _x = _np.arange(Ww, dtype=_np.float64)
        for y in range(Hh):
            s = float(shift_arr[y])
            if abs(s) < 0.25:
                continue
            _src = _x - s
            for c in range(tile.shape[2]):
                out[y, :, c] = _np.interp(_src, _x, tile[y, :, c])
        return out
    # A shift = +dx/2, B shift = -dx/2. dx here is B->A shift: B needs to move -dx to align? We estimated dx as B shift that best matches A.
    # In loop above, we compared A vs B shifted by dx: B[:, dx:] vs A[:, :-dx] means B content at x+dx matches A at x.
    # So B is dx to the right of A. To align to middle: A +dx/2 (right), B -dx/2 (left).
    a2 = _warp_tile(a, dx_full * 0.5, True)
    b2 = _warp_tile(b, -dx_full * 0.5, False)
    return a2, b2, dx_full


def _local_align_in_transition(full_a, full_b, seam_x_arr, ox0, ox1, TH, TW):
    """Local align ONLY inside transition zone |x-seam|<=32, tapered to 0 outside.
    Returns (full_a2, full_b2, dx_arr). Pixels outside zone are bit-identical."""
    import numpy as _np
    import cv2 as _cv2
    ZONE_TOP = 32.0    # <-- top: wall/clock/face, narrow, don't touch far pixels
    ZONE_BOTTOM = 96.0 # <-- bottom: table is wide, needs wider zone (was fixed 48, table front edge extends beyond it)
    MAX_SHIFT = 12.0
    # Estimate dx(y) per band in overlap, same SSD as before, but require improvement
    BAND = 16
    H = full_a.shape[0]
    centres = list(range(BAND // 2, H, BAND))
    dxs = []
    for yc in centres:
        y0 = max(0, yc - 16); y1 = min(H, yc + 16)
        if y1 - y0 < 8:
            dxs.append(0.0); continue
        A = full_a[y0:y1, ox0:ox1].astype(_np.float32)
        B = full_b[y0:y1, ox0:ox1].astype(_np.float32)
        if float(A.std()) < 5.0 and float(B.std()) < 5.0:
            dxs.append(0.0); continue
        # SSD at dx=0 baseline
        _base = float(((A - B) ** 2).mean())
        best_dx, best_ssd = 0, _base
        for dx in range(-12, 13):
            if dx == 0:
                continue
            if dx >= 0:
                if A.shape[1] - dx < 16: continue
                diff = A[:, :A.shape[1] - dx] - B[:, dx:]
            else:
                if A.shape[1] + dx < 16: continue
                diff = A[:, -dx:] - B[:, :B.shape[1] + dx]
            ssd = float((diff * diff).mean())
            if ssd < best_ssd:
                best_ssd, best_dx = ssd, dx
        # Require >10% improvement, else dx=0 (don't warp on bad match - this was the old bug: top dx=8.7 on a wall)
        if best_ssd < 0.90 * _base:
            dxs.append(float(best_dx))
        else:
            dxs.append(0.0)
    dxs = _np.array(dxs, dtype=_np.float64)
    try:
        _xs = _np.array(centres, dtype=_np.float64)
        _ys = _np.arange(H, dtype=_np.float64)
        dx_full = _np.interp(_ys, _xs, dxs)
        k = 5
        pad = _np.pad(dx_full, (k // 2, k // 2), mode="edge")
        dx_full = _np.array([_np.median(pad[i:i + k]) for i in range(len(dx_full))])
        dx_full = _cv2.GaussianBlur(dx_full.astype(_np.float32).reshape(-1, 1), (0, 0), 8.0).reshape(-1).astype(_np.float64)
    except Exception:
        dx_full = _np.interp(_np.arange(H, dtype=_np.float64), _np.array(centres, dtype=_np.float64), dxs)
    dx_full = _np.clip(dx_full, -MAX_SHIFT, MAX_SHIFT)
    # Warp ONLY in zone, tapered: w = cos^2(pi/2 * |x-seam|/ZONE), 1 at seam, 0 at edge
    _xgrid = _np.arange(TW, dtype=_np.float64)[None, :]
    if seam_x_arr is not None:
        _centre = _np.asarray(seam_x_arr, dtype=_np.float64)[:, None]
    else:
        _centre = _np.full((H, 1), 672.0)
    # Per-row zone: top narrow, bottom wide. y=0 -> 32, y=H -> 96. Table (bottom) gets wider processing range.
    _zone_arr = (ZONE_TOP + (ZONE_BOTTOM - ZONE_TOP) * (_np.arange(H, dtype=_np.float64)[:, None] / max(1, H - 1))) * (1.0 - 0.5 * _np.exp(-((_np.arange(H, dtype=_np.float64)[:, None] / max(1, H - 1) - 0.5) / 0.22) ** 2))  # centre NARROWER (fix ghosting)
    _dist = _np.abs(_xgrid - _centre)  # H x TW
    _w = _np.where(_dist <= _zone_arr, _np.cos(_np.pi / 2.0 * _dist / _zone_arr) ** 2, 0.0)  # H x TW
    # Per-row shift: A +dx/2 * w, B -dx/2 * w
    def _warp_zone(img, shift_per_row):
        # shift_per_row: H x TW, shift for each pixel
        out = img.copy()
        _x = _np.arange(TW, dtype=_np.float64)
        for y in range(H):
            _s = shift_per_row[y]  # TW,
            if float(_np.abs(_s).max()) < 0.25:
                continue
            _src = _x - _s
            for c in range(img.shape[2]):
                out[y, :, c] = _np.interp(_src, _x, img[y, :, c])
        return out
    _shift_a = (dx_full[:, None] * 0.5) * _w
    _shift_b = (-dx_full[:, None] * 0.5) * _w
    a2 = _warp_zone(full_a, _shift_a)
    b2 = _warp_zone(full_b, _shift_b)
    return a2, b2, dx_full

def _perform_nvs_tiled_soft_seam(stereo_nvs, orig_pil, target_long, baseline_val, seed_int, get_autocast_fn, seam_mode="auto", seam_x_pct=50.0, _config=None):
    """Tiling with ONE seam: Low=feather, High=winner-take-all at seam.
    - TILE 768, OVERLAP 192, same as before.
    - Each tile diffused at 768 square, resized to tile size.
    - Low band (blurred colour) feather over full overlap -> no colour step.
    - High band (detail) from ONE tile only, cut at seam_x -> no double face/hand.
    - Seam: auto=flattest column, manual=seam_x_pct% inside overlap, off never called here.
    Returns PIL at TW x TH (aspect preserved)."""
    import torch as _torch
    import torchvision.transforms as _transforms
    import torchvision.transforms.functional as _TF
    import numpy as _np
    import cv2 as _cv2
    W0, H0 = orig_pil.size
    scale = float(target_long) / max(W0, H0) if target_long else 1.0
    TW, TH = max(1, int(round(W0 * scale))), max(1, int(round(H0 * scale)))
    # FIX v10.1: Face list from ORIG (not stitched A). Only for head, apple/orange are not faces, so they stay V10.
    _orig_faces = []
    try:
        _orig_small = orig_pil.resize((TW, TH))
        _gray0 = _cv2.cvtColor(_np.asarray(_orig_small).astype(_np.uint8), _cv2.COLOR_RGB2GRAY)
        _cascade0 = _cv2.CascadeClassifier(_cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        _faces0 = _cascade0.detectMultiScale(_gray0, 1.1, 4, minSize=(40, 40))
        for (_fx, _fy, _fw, _fh) in _faces0:
            _orig_faces.append((int(_fx), int(_fy), int(_fw), int(_fh)))
    except Exception:
        _orig_faces = []
    TILE, OVERLAP = 768, 192
    # FIX v21: 传统逻辑错了. 扩散 NVS 里 baseline = 条件信号强度, 不是单纯位移.
    # 0.02: 信号弱, 噪声占上风, 每块各幻觉各的 -> 缝反而大. 0.05: 信号>>噪声, 块对得上 -> 缝最好. 0.10: 信号更强, 人头对得更好, 平图遮挡大 -> 又差.
    # 所以小 baseline 不追小 dx (那是噪声), 大 baseline 才放开搜索.
    try:
        _bl = abs(float(baseline_val)) if baseline_val else 0.05
    except Exception:
        _bl = 0.05
    _SEARCH = int(max(12, min(48, round(400 * _bl))))
    _PEN = float(max(0.5, min(2.5, 1.0 * (0.05 / max(0.01, _bl)))))
    try:
        _has_face_v20 = len(_orig_faces) > 0
    except Exception:
        _has_face_v20 = False
    if _has_face_v20:
        _SEARCH = int(max(_SEARCH, 36))
        _PEN = float(min(_PEN, 0.5))
    if _bl < 0.03:
        _SEARCH = 12
        _PEN = 3.0
    print(f"[SD3D] Baseline {float(baseline_val):.3f} -> stitch search +-{_SEARCH} penalty {_PEN:.2f} face={_has_face_v20}", flush=True)
    # Tile grid in output coords
    xs, x = [], 0
    while x + TILE < TW:
        xs.append(x); x += TILE - OVERLAP
    xs.append(max(0, TW - TILE))
    ys, y = [], 0
    while y + TILE < TH:
        ys.append(y); y += TILE - OVERLAP
    ys.append(max(0, TH - TILE))
    xs, ys = sorted(set(xs)), sorted(set(ys))
    print(f"[SD3D] Resolution {target_long}", flush=True)
    def _diffuse(img_768):
        _to = _transforms.Compose([_transforms.ToTensor(), _transforms.Normalize([0.5], [0.5])])
        _dt = _torch.float16 if _torch.cuda.is_available() else _torch.float32
        _dev = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        _torch.manual_seed(int(seed_int))
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(int(seed_int))
        tb = _to(img_768).to(dtype=_dt, device=_dev).unsqueeze(0)
        b = _torch.full((1, 1), float(baseline_val), device=_dev, dtype=_torch.float32)
        _restore = _set_train_768(stereo_nvs, _config)
        try:
            with get_autocast_fn():
                out = stereo_nvs.perform_nvs(tb, b, intrinsics=None, intrinsics_tgt=None)
        finally:
            try: _restore()
            except: pass
        return _TF.to_pil_image(out[0].detach().float().cpu())
    # Generate each tile
    tiles = []  # (x0,y0,arr)
    for oy in ys:
        for ox in xs:
            tw, th = min(TILE, TW-ox), min(TILE, TH-oy)
            ox0 = int(round(ox*W0/float(TW))); oy0 = int(round(oy*H0/float(TH)))
            ox1 = int(round((ox+tw)*W0/float(TW))); oy1 = int(round((oy+th)*H0/float(TH)))
            tile_orig = orig_pil.crop((max(0,ox0), max(0,oy0), max(ox1,ox0+8), max(oy1,oy0+8)))
            tile_novel = _diffuse(tile_orig.resize((768,768), Image.Resampling.LANCZOS))
            tiles.append((ox, oy, _np.asarray(tile_novel.resize((tw,th), Image.Resampling.LANCZOS)).astype(_np.float64)))
    if len(tiles) == 1:
        return Image.fromarray(_np.clip(tiles[0][2],0,255).astype(_np.uint8))
    # Only handle 2-tile horizontal case with seam (most common: 1344x755). Vertical / 2x2 falls back to feather.
    if len(tiles) == 2 and tiles[0][1] == tiles[1][1]:
        (x0a, y0a, a), (x0b, y0b, b) = tiles[0], tiles[1]
        ox0, ox1 = max(x0a, x0b), min(x0a+TILE, x0b+TILE)
        if seam_mode == "manual":
            pct = max(0.0, min(100.0, float(seam_x_pct)))
            seam_x = int(ox0 + (ox1-ox0)*pct/100.0)
            seam_x_arr = None  # manual = straight line you asked for
            print(f"[SD3D] Seam: manual", flush=True)
        else:
            # Auto: WAVY seam - one x per row, snakes around objects. Not one straight line.
            orig_target = orig_pil.resize((TW, TH), Image.Resampling.LANCZOS)
            _wavy = _find_wavy_seam(orig_target, ox0, ox1)
            if _wavy is not None and len(_wavy) == TH:
                seam_x_arr = _wavy  # per-row, shape (TH,)
                seam_x = int(round(float(_wavy.mean())))
                pass  # [SD3D] log suppressed (Easy3D-style)
            else:
                seam_x_arr = None
                seam_x = _find_softest_seam(orig_target, ox0, ox1)
                pass  # [SD3D] log suppressed (Easy3D-style)
        # PER-ROW ALIGN DISABLED 2026-11-14: it warped the WHOLE tile (not just overlap)
        # and SSD-matched two different AI views -> top dx=8.7 (should be ~0), whole right half wavy-smeared.
        # Keep tiles as-is. Seam only moves the seam line, never the pixels.
        # If you want it back later, it must be: warp ONLY inside overlap, taper to 0 at 96px from seam, |dx|<=8, and require SSD improvement.
        pass  # [SD3D] log suppressed (Easy3D-style)
        # Build full canvas: ONE smoothstep alpha blend across wavy seam (no low/high split).
        # Why old code had no transition: low was linear feather over 192px, high was linear ramp over 96px,
        # two different widths + GaussianBlur on zero-padded canvas (dark halo at tile edges) + linear ramp has kinks at both ends.
        # Eye sees the kink as a hard line even when edges are aligned.
        # Fix: single alpha, smoothstep (3t^2-2t^3), TRANSITION=64px centred on wavy seam_x(y).
        # - alpha=1 left of seam (pure A), alpha=0 right (pure B), smooth S-curve in between, no kink, sum=1 so no brightness change.
        # - No blur, no warping, no pixel moving - only blending weights change.
        # Place tiles on canvas
        full_a = _np.zeros((TH, TW, 3)); full_b = _np.zeros((TH, TW, 3))
        full_a[y0a:y0a+a.shape[0], x0a:x0a+a.shape[1]] = a
        full_b[y0b:y0b+b.shape[0], x0b:x0b+b.shape[1]] = b
        # LOCAL ALIGN: only inside transition zone, tapered to 0 outside. Fixes bottom disparity without smearing whole tile.
        try:
            full_a, full_b, _dx_local = _local_align_in_transition(full_a, full_b, seam_x_arr if 'seam_x_arr' in locals() else None, ox0, ox1, TH, TW) if abs(float(baseline_val)) >= 0.03 else (full_a, full_b, __import__('numpy').zeros((TH,)))
            _top = float(_dx_local[:max(1, TH // 4)].mean()); _mid = float(_dx_local[TH // 4:TH * 3 // 4].mean()); _bot = float(_dx_local[TH * 3 // 4:].mean())
            print(f"[SD3D] Aligning seams", flush=True)
        except Exception as _lae:
            pass  # [SD3D] log suppressed (Easy3D-style)
        TRANSITION = 64.0  # <-- knob: 32=sharper, 64=default, 96=softer (more ghost if misaligned, but you said edges are aligned)
        _xs = _np.arange(TW, dtype=_np.float64)[None, :, None]  # 1 x TW x 1
        if 'seam_x_arr' in locals() and seam_x_arr is not None:
            _centre = _np.asarray(seam_x_arr, dtype=_np.float64)[:, None, None]  # TH x 1 x 1, wavy per-row
        else:
            _centre = float(seam_x)
        # t = 0 at left edge of transition, 1 at right edge. Outside: <0 -> alpha=1, >1 -> alpha=0.
        _trans_arr = (48.0 + (96.0 - 48.0) * (_np.arange(TH, dtype=_np.float64)[:, None, None] / max(1, TH - 1))) * (1.0 - 0.5 * _np.exp(-((_np.arange(TH, dtype=_np.float64)[:, None, None] / max(1, TH - 1) - 0.5) / 0.22) ** 2))
        _tt = (_xs - (_centre - _trans_arr / 2.0)) / _trans_arr
        _tt = _np.clip(_tt, 0.0, 1.0)
        # Smoothstep: 3t^2 - 2t^3, derivative 0 at both ends -> no kink, no hard line.
        _s = _tt * _tt * (3.0 - 2.0 * _tt)
        _alpha = 1.0 - _s  # 1 = A, 0 = B
        # Outside overlap, force pure covering tile (avoid blending with zero-padded canvas)
        _alpha = _np.where(_xs < ox0, 1.0, _alpha)
        _alpha = _np.where(_xs >= ox1, 0.0, _alpha)
        # Broadcast alpha to TH x TW x 3 if seam is straight (scalar centre) - _alpha is 1xTWx1, need TH
        if _alpha.shape[0] == 1:
            _alpha = _np.repeat(_alpha, TH, axis=0)
        out = full_a * _alpha + full_b * (1.0 - _alpha)
        pass  # [SD3D] log suppressed (Easy3D-style)
        return Image.fromarray(_np.clip(out,0,255).astype(_np.uint8))
    # Generic 2D stitch (ALL GUI sizes) - WAVY CUT, not feather.
    # Root cause of centre ghosting: two tiles hallucinate DIFFERENT pillar/face positions (10-20px apart).
    # Feather blends them -> double pillar. Narrow feather still blends -> still double.
    # Fix: for each seam, DP finds per-row (vertical seam) / per-col (horizontal seam) cut that
    # minimises |A-B| + smoothness, i.e. cuts through flat sky/cloud/wall, NOT through pillar/face.
    # Then hard cut: left side = A only, right side = B only, only 8px smoothstep around cut to hide line.
    # 2x2: top = cut(top-left, top-right), bottom = cut(bottom-left, bottom-right), final = cut(top, bottom) in Y.
    # 3x3 / 4x4: same pairwise, left-to-right then top-to-bottom. No feather, no ghosting.
    print(f"[SD3D] Stitching", flush=True)
    import cv2 as _cv2g
    def _cut_h(a, b, ox0, ox1):
        # a,b: TH x TW x 3 full-canvas images (zero outside tile), ox0..ox1 overlap in X
        # Returns TH x TW x 3, left of seam = a, right = b
        # FIX: GLOBAL SHIFT first. Whole block may be 10-30px off, not just seam.
        # Estimate dx over WHOLE overlap (not just seam), shift entire B by -dx, then cut.
        H, W = a.shape[:2]
        # FIX v9: STEP (断层), not bend. Two tiles' diagonal lines have different slopes, translation alone leaves a step.
        # Per-band dx bends lines (kinks at band borders). Shear dx(y)=p+q*y keeps straight lines straight, fixes slope + step.
        # Faces: don't warp hard, just 32px horizontal smear below (child head only needs slight smear).
        try:
            _lo, _hi = max(0, ox0), min(W, ox1)
            _ga_full = _cv2g.cvtColor(_np.clip(a, 0, 255).astype(_np.uint8), _cv2g.COLOR_RGB2GRAY).astype(_np.float64)
            _gb_full = _cv2g.cvtColor(_np.clip(b, 0, 255).astype(_np.uint8), _cv2g.COLOR_RGB2GRAY).astype(_np.float64)
            # FIX v10: EDGE-weighted, not gray SSD. Gray is dominated by wood/fabric texture, lines get drowned.
            # Sobel magnitude: lines/edges are bright, flat texture is dark, so dx is fitted to lines.
            _gx_a = _cv2g.Sobel(_ga_full, _cv2g.CV_64F, 1, 0, ksize=3)
            _gy_a = _cv2g.Sobel(_ga_full, _cv2g.CV_64F, 0, 1, ksize=3)
            _ga_e = _np.sqrt(_gx_a**2 + _gy_a**2)
            _gx_b = _cv2g.Sobel(_gb_full, _cv2g.CV_64F, 1, 0, ksize=3)
            _gy_b = _cv2g.Sobel(_gb_full, _cv2g.CV_64F, 0, 1, ksize=3)
            _gb_e = _np.sqrt(_gx_b**2 + _gy_b**2)
            # Estimate dx for top half and bottom half, fit linear dx(y)
            _dx_halves = []
            for _y0, _y1 in [(0, H // 2), (H // 2, H)]:
                _best_s, _best_ssd = 0, 1e18
                for _s in range(-_SEARCH, _SEARCH + 1, 1):
                    _M0 = _np.float32([[1, 0, _s], [0, 1, 0]])
                    _gb_s = _cv2g.warpAffine(_gb_e, _M0, (W, H), flags=_cv2g.INTER_LINEAR, borderMode=_cv2g.BORDER_REPLICATE)
                    _ssd = _np.mean((_ga_e[_y0:_y1, _lo:_hi][:, 5:-5] - _gb_s[_y0:_y1, _lo:_hi][:, 5:-5]) ** 2)
                    _ssd = _ssd + _PEN * abs(_s)
                    if _ssd < _best_ssd:
                        _best_ssd, _best_s = _ssd, _s
                _dx_halves.append(float(_best_s))
            _dx_top, _dx_bot = _dx_halves[0], _dx_halves[1]
            # Linear shear: dx(y) = dx_top + (dx_bot-dx_top)*(y/H). Straight stays straight.
            _ys = _np.arange(H, dtype=_np.float64)
            _dx_y = _dx_top + (_dx_bot - _dx_top) * (_ys / max(1, H - 1))
            # FIX v10.1: Face covers overlap -> NO warp (dx=0), head from one tile, no upper/lower split. Apple/orange not faces, keep shear.
            try:
                for (_fx, _fy, _fw, _fh) in _orig_faces:
                    if _fx <= _lo and _fx + _fw >= _hi and _fy <= H // 2 <= _fy + _fh:
                        _dx_y = _np.zeros_like(_dx_y)
                        break
            except Exception:
                pass
            _xs_grid, _ys_grid = _np.meshgrid(_np.arange(W, dtype=_np.float32), _np.arange(H, dtype=_np.float32))
            _map_x = (_xs_grid - _dx_y[:, None].astype(_np.float32)).astype(_np.float32)
            _map_y = _ys_grid.astype(_np.float32)
            b = _cv2g.remap(b, _map_x, _map_y, interpolation=_cv2g.INTER_LINEAR, borderMode=_cv2g.BORDER_REPLICATE)
        except Exception:
            pass
        # FIX v10.1: Face covers overlap -> seam at edge, not mid. Head from ONE tile, not split into left/right.
        # Face centre left of overlap centre -> seam at right edge (head from left tile). Else seam at left edge.
        # No warp for face (dx already 0 if face, see below). Apple/orange are not faces, so they keep V10 mid seam.
        # STRAIGHT seam + 32px horizontal smear (not DP wavy seam - DP bends lines)
        _lo2, _hi2 = max(0, ox0), min(W, ox1)
        _mid = (_lo2 + _hi2) // 2
        try:
            for (_fx, _fy, _fw, _fh) in _orig_faces:
                if _fx <= _lo2 and _fx + _fw >= _hi2 and _fy <= H // 2 <= _fy + _fh:
                    _fcx = (_fx + _fx + _fw) / 2.0
                    _ocx = (_lo2 + _hi2) / 2.0
                    _mid = int(_hi2 - 2) if _fcx < _ocx else int(_lo2 + 2)
                    break
        except Exception:
            pass
        _xs = _np.arange(W, dtype=_np.float64)[None, :]
        _tt = _np.clip((_xs - (_mid - 16.0)) / 32.0, 0.0, 1.0)
        _s = _tt * _tt * (3.0 - 2.0 * _tt)
        _alpha = 1.0 - _s  # 1 left (a), 0 right (b)
        _alpha[:, :_lo2] = 1.0
        _alpha[:, _hi2:] = 0.0
        return a * _alpha[:, :, None] + b * (1.0 - _alpha[:, :, None])

    def _cut_v(a, b, oy0, oy1):
        # Vertical cut: top = a, bottom = b, seam per column
        # FIX: GLOBAL SHIFT in Y, same as _cut_h. Whole block may be 10-30px off in Y.
        H, W = a.shape[:2]
        # FIX v9 for V: shear dx(x)=p+q*x, keeps straight lines straight, fixes step across horizontal seam.
        try:
            _lo, _hi = max(0, oy0), min(H, oy1)
            _ga_full = _cv2g.cvtColor(_np.clip(a, 0, 255).astype(_np.uint8), _cv2g.COLOR_RGB2GRAY).astype(_np.float64)
            _gb_full = _cv2g.cvtColor(_np.clip(b, 0, 255).astype(_np.uint8), _cv2g.COLOR_RGB2GRAY).astype(_np.float64)
            _gx_a = _cv2g.Sobel(_ga_full, _cv2g.CV_64F, 1, 0, ksize=3)
            _gy_a = _cv2g.Sobel(_ga_full, _cv2g.CV_64F, 0, 1, ksize=3)
            _ga_e = _np.sqrt(_gx_a**2 + _gy_a**2)
            _gx_b = _cv2g.Sobel(_gb_full, _cv2g.CV_64F, 1, 0, ksize=3)
            _gy_b = _cv2g.Sobel(_gb_full, _cv2g.CV_64F, 0, 1, ksize=3)
            _gb_e = _np.sqrt(_gx_b**2 + _gy_b**2)
            _dx_halves = []
            for _x0, _x1 in [(0, W // 2), (W // 2, W)]:
                _best_s, _best_ssd = 0, 1e18
                for _s in range(-_SEARCH, _SEARCH + 1, 1):
                    _M0 = _np.float32([[1, 0, _s], [0, 1, 0]])
                    _gb_s = _cv2g.warpAffine(_gb_e, _M0, (W, H), flags=_cv2g.INTER_LINEAR, borderMode=_cv2g.BORDER_REPLICATE)
                    _ssd = _np.mean((_ga_e[_lo:_hi, _x0:_x1][5:-5, :] - _gb_s[_lo:_hi, _x0:_x1][5:-5, :]) ** 2)
                    _ssd = _ssd + _PEN * abs(_s)
                    if _ssd < _best_ssd:
                        _best_ssd, _best_s = _ssd, _s
                _dx_halves.append(float(_best_s))
            _dx_left, _dx_right = _dx_halves[0], _dx_halves[1]
            _xs = _np.arange(W, dtype=_np.float64)
            _dx_x = _dx_left + (_dx_right - _dx_left) * (_xs / max(1, W - 1))
            try:
                for (_fx, _fy, _fw, _fh) in _orig_faces:
                    if _fy <= _lo and _fy + _fh >= _hi and _fx <= W // 2 <= _fx + _fw:
                        _dx_x = _np.zeros_like(_dx_x)
                        break
            except Exception:
                pass
            _xs_grid, _ys_grid = _np.meshgrid(_np.arange(W, dtype=_np.float32), _np.arange(H, dtype=_np.float32))
            _map_x = (_xs_grid - _dx_x[None, :].astype(_np.float32)).astype(_np.float32)
            _map_y = _ys_grid.astype(_np.float32)
            b = _cv2g.remap(b, _map_x, _map_y, interpolation=_cv2g.INTER_LINEAR, borderMode=_cv2g.BORDER_REPLICATE)
        except Exception:
            pass
        _lo2, _hi2 = max(0, oy0), min(H, oy1)
        _mid = (_lo2 + _hi2) // 2
        try:
            for (_fx, _fy, _fw, _fh) in _orig_faces:
                if _fy <= _lo2 and _fy + _fh >= _hi2 and _fx <= W // 2 <= _fx + _fw:
                    _fcy = (_fy + _fy + _fh) / 2.0
                    _ocy = (_lo2 + _hi2) / 2.0
                    _mid = int(_hi2 - 2) if _fcy < _ocy else int(_lo2 + 2)
                    break
        except Exception:
            pass
        _ys = _np.arange(H, dtype=_np.float64)[:, None]
        _tt = _np.clip((_ys - (_mid - 16.0)) / 32.0, 0.0, 1.0)
        _s = _tt * _tt * (3.0 - 2.0 * _tt)
        _alpha = 1.0 - _s
        _alpha[:_lo2, :] = 1.0
        _alpha[_hi2:, :] = 0.0
        return a * _alpha[:, :, None] + b * (1.0 - _alpha[:, :, None])

    # Build full-canvas images per tile
    _fulls = []
    for (ox, oy, arr) in tiles:
        _f = _np.zeros((TH, TW, 3))
        _f[oy:oy+arr.shape[0], ox:ox+arr.shape[1]] = arr
        _fulls.append(((ox, oy, ox+arr.shape[1], oy+arr.shape[0]), _f))
    # Group by row (same oy) -> stitch horizontally, then stitch rows vertically
    from collections import defaultdict as _dd
    _rows = _dd(list)
    for _b, _f in _fulls:
        _rows[_b[1]].append((_b, _f))
    _row_imgs = []
    _row_bounds = []
    for _oy in sorted(_rows.keys()):
        _lst = sorted(_rows[_oy], key=lambda x: x[0][0])
        _cur_b, _cur = _lst[0][0], _lst[0][1]
        for _nb, _nf in _lst[1:]:
            _ox0, _ox1 = _nb[0], _cur_b[2]  # overlap start = next tile x0, end = cur tile x1
            _cur = _cut_h(_cur, _nf, _ox0, _ox1)
            _cur_b = (_cur_b[0], _cur_b[1], max(_cur_b[2], _nb[2]), max(_cur_b[3], _nb[3]))
        _row_imgs.append(_cur)
        _row_bounds.append(_cur_b)
    _out = _row_imgs[0]
    _cur_b = _row_bounds[0]
    for _nxt, _nb in zip(_row_imgs[1:], _row_bounds[1:]):
        _oy0, _oy1 = _nb[1], _cur_b[3]
        _out = _cut_v(_out, _nxt, _oy0, _oy1)
        _cur_b = (_cur_b[0], _cur_b[1], max(_cur_b[2], _nb[2]), max(_cur_b[3], _nb[3]))
    out = _out
    return Image.fromarray(_np.clip(out, 0, 255).astype(_np.uint8))

def generate_novel_view(args, config, stereo_nvs=None):
    from src import StereoSpace
    gamma_a_val = float(getattr(args, 'gamma_a', 1.0) or 1.0)
    gamma_b_val = float(getattr(args, 'gamma_b', 1.0) or 1.0)
    legacy_gamma = float(getattr(args, 'gamma', 1.0) or 1.0)
    if abs(legacy_gamma - 1.0) > 0.001 and abs(gamma_b_val - 1.0) < 0.001:
        gamma_b_val = legacy_gamma
        pass  # [SD3D] log suppressed (Easy3D-style)
    pass  # [SD3D] log suppressed (Easy3D-style)
    K_src = torch.tensor(args.src_intrinsics, dtype=torch.float32).reshape(3, 3).to(DEVICE) if args.src_intrinsics else None
    K_tgt = torch.tensor(args.tgt_intrinsics, dtype=torch.float32).reshape(3, 3).to(DEVICE) if args.tgt_intrinsics else None
    os.makedirs(args.output, exist_ok=True)
    image_paths = collect_image_paths(args.input)
    # === FIX V80.12: 优先使用 args.high_res，忽略 config 限制 ===
    requested_res = int(getattr(args, 'high_res', 0) or 0)
    if requested_res > 0:
        crop_size = requested_res
    else:
        crop_size = int(config.data.train_width)
    print(f"[SD3D] Resolution {crop_size}", flush=True)
    if stereo_nvs is None:
        stereo_nvs = StereoSpace(config, DEVICE)
        try:
            pipe = getattr(stereo_nvs, 'pipe', None) or getattr(stereo_nvs, 'pipeline', None)
            if pipe:
                if hasattr(pipe, 'unet'):
                    try:
                        pipe.unet.enable_xformers_memory_efficient_attention()
                        print("[V80.9] xformers ON")
                    except: pass
                try:
                    from diffusers import DPMSolverMultistepScheduler
                    if hasattr(pipe, 'scheduler'):
                        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
                        print(f"[V80.9] DPM++ {config.num_inference_steps} steps")
                except: pass
        except: pass
    # --- Seam Tiling branch (V75.36): if seam_mode != off and target>768, tile with soft/manual seam, save SBS, return ---
    # This is the Sisi fix: seam no longer at mid-overlap blindly, it goes to softest column or your manual %.
    _seam_mode = str(getattr(args, "seam_mode", "auto") or "auto").lower()
    _seam_x = float(getattr(args, "seam_x", 50.0) or 50.0)
    _req = int(getattr(args, "high_res", 0) or 0)
    if _seam_mode in ("auto", "manual") and _req and _req > 768:
        try:
            from os.path import basename as _bn, splitext as _sp
            _paths = collect_image_paths(args.input)
            _seed0 = int(args.seed) if args.seed is not None else 42
            _sym = args.mode in ("both_generated", "symmetric", "both")
            for _p in _paths:
                _orig = Image.open(_p).convert("RGB")
                _name = _sp(_bn(_p))[0]
                if _sym:
                    _l = _perform_nvs_tiled_soft_seam(stereo_nvs, _orig, _req, -args.baseline/2.0, _seed0, get_autocast, seam_mode=_seam_mode, seam_x_pct=_seam_x, _config=config)
                    _r = _perform_nvs_tiled_soft_seam(stereo_nvs, _orig, _req, args.baseline/2.0, _seed0, get_autocast, seam_mode=_seam_mode, seam_x_pct=_seam_x, _config=config)
                    _sbs = create_sbs_image(_l, _r, getattr(args, "output_type", "parallel"))
                    _sp_path = join(args.output, f"{_name}_SD3D_S{int(getattr(config, 'num_inference_steps', 8) or 8)}_R{_req}.3D_SBS_B{args.baseline:.2f}_S{_seed0}.jpg")
                else:
                    _b = -args.baseline if args.mode == "orig_left" else args.baseline
                    _syn = _perform_nvs_tiled_soft_seam(stereo_nvs, _orig, _req, _b, _seed0, get_autocast, seam_mode=_seam_mode, seam_x_pct=_seam_x, _config=config)
                    if args.mode == "orig_left":
                        _sbs = create_sbs_image(_syn, _orig.resize(_syn.size, Image.Resampling.LANCZOS), getattr(args, "output_type", "parallel"))
                        _sp_path = join(args.output, f"{_name}_L_SD3D_S{int(getattr(config, 'num_inference_steps', 8) or 8)}_R{_req}.3D_SBS_B{args.baseline:.2f}_S{_seed0}.jpg")
                    else:
                        _sbs = create_sbs_image(_orig.resize(_syn.size, Image.Resampling.LANCZOS), _syn, getattr(args, "output_type", "parallel"))
                        _sp_path = join(args.output, f"{_name}_R_SD3D_S{int(getattr(config, 'num_inference_steps', 8) or 8)}_R{_req}.3D_SBS_B{args.baseline:.2f}_S{_seed0}.jpg")
                _sbs.save(_sp_path, "JPEG", quality=95, subsampling=0)
                print(f"[SD3D] Saved: {_bn(_sp_path)}", flush=True)
            print(f"[SD3D] Done", flush=True)
            return
        except Exception as _te:
            print(f"[SD3D] Failed, retrying", flush=True)
            import traceback; traceback.print_exc()
    bs = max(1, int(args.batch_size))
    symmetric = args.mode in ("both_generated", "symmetric", "both")
    model_type, target_long, key = parse_enhance(args.enhance)
    # V89: skip SDXL preload for 0.05/0.10 to avoid OOM and keep fast
    clarity_tmp = getattr(args, 'clarity', 'none')
    is_low_tmp = False
    try:
        if "0.05" in str(clarity_tmp) or "0.10" in str(clarity_tmp):
            is_low_tmp = True
    except:
        pass
    if model_type and not is_low_tmp:
        try:
            from enhance import init_enhance_model
            init_enhance_model(model_type)
        except Exception as e:
            print(f"[Enhance] Init failed {e}")
    seed_int = int(args.seed) if args.seed is not None else 42
    random.seed(seed_int)
    np.random.seed(seed_int)
    torch.manual_seed(seed_int)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_int)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[SEED] Fixed seed={seed_int} - random/numpy/torch/cuda all fixed - V84 np imported")
    if args.face_restore != "none":
        init_face_restorer(restore_type=args.face_restore, fidelity=args.face_fidelity)
    print(f"[MODE] {args.mode} baseline {args.baseline}")
    for start in range(0, len(image_paths), bs):
        batch_seed = seed_int + start
        random.seed(batch_seed)
        np.random.seed(batch_seed)
        torch.manual_seed(batch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(batch_seed)
        print(f"[SEED] Batch {start//bs} seed={batch_seed}")
        batch_paths = image_paths[start : start + bs]
        src_list = []
        intr_src_list = []
        intr_tgt_list = []
        names = []
        letterbox_list = []
        orig_list = []
        orig_size_list = []
        for p in batch_paths:
            src_1, intr_src_1, intr_tgt_1, base_name, letterbox, orig_pil, orig_size = process_image(p, K_src, K_tgt, crop_size=crop_size, resize_mode=args.resize_mode, gamma_a=gamma_a_val)
            src_list.append(src_1)
            intr_src_list.append(intr_src_1)
            intr_tgt_list.append(intr_tgt_1)
            names.append(base_name)
            letterbox_list.append(letterbox)
            orig_list.append(orig_pil)
            orig_size_list.append(orig_size)
        src_batch = torch.cat(src_list, dim=0)
        if intr_src_list[0] is not None:
            intr_src_batch = torch.cat(intr_src_list, dim=0) if intr_src_list[0].dim()==3 else torch.stack(intr_src_list)
        else:
            intr_src_batch = None
        if intr_tgt_list[0] is not None:
            intr_tgt_batch = torch.cat(intr_tgt_list, dim=0) if intr_tgt_list[0].dim()==3 else torch.stack(intr_tgt_list)
        else:
            intr_tgt_batch = None
        if symmetric:
            baseline_left = torch.full((src_batch.size(0), 1), float(-args.baseline / 2.0), device=DEVICE, dtype=torch.float32)
            baseline_right = torch.full((src_batch.size(0), 1), float(args.baseline / 2.0), device=DEVICE, dtype=torch.float32)
            with get_autocast():
                synthesized_left = stereo_nvs.perform_nvs(src_batch, baseline_left, intrinsics=intr_src_batch, intrinsics_tgt=intr_tgt_batch)
                synthesized_right = stereo_nvs.perform_nvs(src_batch, baseline_right, intrinsics=intr_src_batch, intrinsics_tgt=intr_tgt_batch)
            synthesized = synthesized_right
        elif args.mode == "orig_left":
            baseline = torch.full((src_batch.size(0), 1), float(-args.baseline), device=DEVICE, dtype=torch.float32)
            with get_autocast():
                synthesized = stereo_nvs.perform_nvs(src_batch, baseline, intrinsics=intr_src_batch, intrinsics_tgt=intr_tgt_batch)
            synthesized_left = None
            synthesized_right = None
        else:
            baseline = torch.full((src_batch.size(0), 1), float(args.baseline), device=DEVICE, dtype=torch.float32)
            with get_autocast():
                synthesized = stereo_nvs.perform_nvs(src_batch, baseline, intrinsics=intr_src_batch, intrinsics_tgt=intr_tgt_batch)
            synthesized_left = None
            synthesized_right = None
        for i, name in enumerate(names):
            left, top, new_w, new_h = letterbox_list[i]
            orig_pil = orig_list[i]
            orig_w, orig_h = orig_size_list[i]
            if symmetric:
                out_left = synthesized_left[i].detach().float().cpu()
                out_right = synthesized_right[i].detach().float().cpu()
                syn_left_pil = TF_F.to_pil_image(out_left)
                syn_right_pil = TF_F.to_pil_image(out_right)
                syn_left_cropped = syn_left_pil.crop((left, top, left+new_w, top+new_h))
                syn_right_cropped = syn_right_pil.crop((left, top, left+new_w, top+new_h))
                syn_l_hq = syn_left_cropped.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
                syn_r_hq = syn_right_cropped.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
                if args.face_restore != "none":
                    blend = 0.5 if "half" in args.face_restore else 1.0
                    syn_l_hq = restore_face(syn_l_hq, blend=blend)
                    syn_r_hq = restore_face(syn_r_hq, blend=blend)
                if args.text_fix > 0:
                    syn_l_hq = preserve_text_regions(orig_pil, syn_l_hq, strength=args.text_fix, baseline=args.baseline/2, mode=args.mode)
                    syn_r_hq = preserve_text_regions(orig_pil, syn_r_hq, strength=args.text_fix, baseline=args.baseline/2, mode=args.mode)
                if args.line_fix > 0:
                    syn_l_hq = preserve_line_regions(orig_pil, syn_l_hq, strength=args.line_fix, baseline=args.baseline/2, mode=args.mode)
                    syn_r_hq = preserve_line_regions(orig_pil, syn_r_hq, strength=args.line_fix, baseline=args.baseline/2, mode=args.mode)
                # V75.21 Gamma B symmetric Both Eyes per-eye AFTER Face
                if abs(gamma_b_val-1.0)>0.001:
                    print(f"  [Gamma B] V75.21 {gamma_b_val} to BOTH eyes symmetric AFTER Face - final", flush=True)
                    syn_l_hq = apply_gamma(syn_l_hq, gamma_b_val)
                    syn_r_hq = apply_gamma(syn_r_hq, gamma_b_val)
                # V74.22 Clarity Global Both Eyes for symmetric mode
                clarity_arg = getattr(args, 'clarity', 'none')
                if clarity_arg and clarity_arg.lower() != "none":
                    try:
                        parts = str(clarity_arg).split('_')
                        cs = float(parts[0])
                        cm = '_'.join(parts[1:]) if len(parts) > 1 else "SBS_Ref_All"
                        from enhance import apply_enhance as apply_clarity
                        fct = f"clarity_{cs}_{cm}"
                        cb = getattr(args, 'clarity_blend', 'none')
                        print(f"  [CLARITY] V105.1 GLOBAL SYMMETRIC {cs} {cm} blend={cb}", flush=True)
                        syn_l_hq = apply_clarity(syn_l_hq, 1.0, model_type=fct, antialias=0, seed=seed_int + i, clarity_blend=cb)
                        syn_r_hq = apply_clarity(syn_r_hq, 1.0, model_type=fct, antialias=0, seed=seed_int + i, clarity_blend=cb)
                        print(f"  [Clarity] V105.1 GLOBAL SYMMETRIC DONE", flush=True)
                    except Exception as e:
                        print(f"  [Clarity] Symmetric failed {e}", flush=True)
                if target_long:
                    try:
                        from enhance import smart_enhance
                        syn_l_hq = smart_enhance(syn_l_hq, target_long, model_type=model_type, antialias=0, seed=seed_int)
                        syn_r_hq = smart_enhance(syn_r_hq, target_long, model_type=model_type, antialias=0, seed=seed_int)
                    except Exception as e:
                        print(f"[Enhance] Failed {e}")
                # V75.30 Use output_type.py for SBS creation
                sbs_image = create_sbs_image(syn_l_hq, syn_r_hq, getattr(args, "output_type", "parallel"))
                face_type = args.face_restore
                suffix = f"_{face_type.upper()}" if "none" not in face_type else ""
                if target_long:
                    suffix += f"_{model_type.upper()}{key.upper()}"
                # V89 FIX: restore Gamma A/B and Clarity suffix
                if abs(gamma_a_val-1.0) > 0.001:
                    suffix += f"_GA{gamma_a_val:.2f}"
                if abs(gamma_b_val-1.0) > 0.001:
                    suffix += f"_GB{gamma_b_val:.2f}"
                clarity_arg = getattr(args, 'clarity', 'none')
                if clarity_arg and clarity_arg.lower() != "none":
                    c_clean = str(clarity_arg).replace(".", "p").replace("_", "").replace("-", "").upper()
                    suffix += f"_{c_clean}"
                if abs(args.antialias) > 0.01:
                    suffix += f"_AA{args.antialias:.1f}"
                if _is_surface_blur_number(args.surface_blur):
                    suffix += f"_BLUR{_get_surface_blur_float(args.surface_blur)}"
                if args.text_fix > 0:
                    suffix += f"_TEXT{args.text_fix:.1f}"
                if args.line_fix > 0:
                    suffix += f"_LINE{args.line_fix:.1f}"
                sbs_path = join(args.output, f"{name}{suffix}.3D_SBS_B{args.baseline:.2f}_S{seed_int}.jpg")
            else:
                out_syn = synthesized[i].detach().float().cpu()
                syn_pil = TF_F.to_pil_image(out_syn)
                syn_cropped = syn_pil.crop((left, top, left+new_w, top+new_h))
                syn_hq = syn_cropped.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
                if orig_pil.size != (orig_w, orig_h):
                    orig_pil = orig_pil.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
                clarity_arg = getattr(args, 'clarity', 'none')
                clarity_strength = 0.0
                clarity_mode = "None"
                if clarity_arg and clarity_arg.lower() != "none":
                    try:
                        parts = str(clarity_arg).split('_')
                        clarity_strength = float(parts[0])
                        clarity_mode = '_'.join(parts[1:]) if len(parts) > 1 else "SBS_Ref_All"
                    except:
                        clarity_strength = 0.0
                if clarity_strength and clarity_strength > 0:
                    try:
                        from enhance import apply_enhance as apply_clarity
                        full_clarity_type = f"clarity_{clarity_strength}_{clarity_mode}" if clarity_mode else f"clarity_{clarity_strength}"
                        # V74.22 GLOBAL BOTH EYES - clarity_blend from args
                        clarity_blend_arg = getattr(args, 'clarity_blend', 'none')
                        print(f"  [CLARITY] V105.1 GLOBAL BOTH EYES {clarity_strength} {clarity_mode} blend={clarity_blend_arg} -> {full_clarity_type} seed {seed_int + i}", flush=True)
                        # Apply to BOTH eyes - global
                        syn_hq = apply_clarity(syn_hq, 1.0, model_type=full_clarity_type, antialias=0, seed=seed_int + i, clarity_blend=clarity_blend_arg)
                        orig_pil = apply_clarity(orig_pil, 1.0, model_type=full_clarity_type, antialias=0, seed=seed_int + i, clarity_blend=clarity_blend_arg)
                        print(f"  [Clarity] V105.1 GLOBAL DONE {full_clarity_type} both eyes", flush=True)
                    except Exception as e:
                        print(f"  [Clarity] V105.1 failed {e}", flush=True)
                        import traceback; traceback.print_exc()
                if args.face_restore != "none":
                    blend = 0.5 if "half" in args.face_restore else 1.0
                    if "both" in args.face_restore:
                        orig_pil = restore_face(orig_pil, blend=blend)
                        syn_hq = restore_face(syn_hq, blend=blend)
                    else:
                        syn_hq = restore_face(syn_hq, blend=blend)
                # V75.20 Gamma B per-eye AFTER Face, so Gamma B actually darkens final output (fix preview only)
                if abs(gamma_b_val-1.0)>0.001:
                    print(f"  [Gamma B] V75.20 {gamma_b_val} to BOTH eyes per-eye AFTER Face - final dark", flush=True)
                    syn_hq = apply_gamma(syn_hq, gamma_b_val)
                    orig_pil = apply_gamma(orig_pil, gamma_b_val)
                if args.text_fix > 0:
                    syn_hq = preserve_text_regions(orig_pil, syn_hq, strength=args.text_fix, baseline=args.baseline, mode=args.mode)
                if args.line_fix > 0:
                    syn_hq = preserve_line_regions(orig_pil, syn_hq, strength=args.line_fix, baseline=args.baseline, mode=args.mode)
                if _is_surface_blur_number(args.surface_blur):
                    syn_hq = apply_surface_blur(syn_hq, _get_surface_blur_float(args.surface_blur))
                if args.antialias > 0.01:
                    orig_pil = apply_antialias(orig_pil, args.antialias)
                    syn_hq = apply_antialias(syn_hq, args.antialias)
                if target_long:
                    try:
                        from enhance import smart_enhance
                        syn_hq = smart_enhance(syn_hq, target_long, model_type=model_type, antialias=0, seed=seed_int)
                        orig_pil_enh = smart_enhance(orig_pil, target_long, model_type=model_type, antialias=0, seed=seed_int)
                    except Exception as e:
                        print(f"[Enhance] Failed {e}")
                        orig_pil_enh = orig_pil
                else:
                    orig_pil_enh = orig_pil
                if args.mode == "orig_left":
                    # V75.30 Use output_type.py - Left AI, Right Orig
                    sbs_image = create_sbs_image(syn_hq, orig_pil_enh, getattr(args, "output_type", "parallel"))
                    suffix = f"_{args.face_restore.upper()}" if "none" not in args.face_restore else ""
                    if target_long:
                        suffix += f"_{model_type.upper()}{key.upper()}"
                    # V89 FIX: restore Gamma A/B and Clarity suffix
                    if abs(gamma_a_val-1.0) > 0.001:
                        suffix += f"_GA{gamma_a_val:.2f}"
                    if abs(gamma_b_val-1.0) > 0.001:
                        suffix += f"_GB{gamma_b_val:.2f}"
                    clarity_arg = getattr(args, 'clarity', 'none')
                    if clarity_arg and clarity_arg.lower() != "none":
                        c_clean = str(clarity_arg).replace(".", "p").replace("_", "").replace("-", "").upper()
                        suffix += f"_{c_clean}"
                    if abs(args.antialias) > 0.01:
                        suffix += f"_AA{args.antialias:.1f}"
                    if _is_surface_blur_number(args.surface_blur):
                        suffix += f"_BLUR{_get_surface_blur_float(args.surface_blur)}"
                    if args.text_fix > 0:
                        suffix += f"_TEXT{args.text_fix:.1f}"
                    if args.line_fix > 0:
                        suffix += f"_LINE{args.line_fix:.1f}"
                    sbs_path = join(args.output, f"{name}_L{suffix}.3D_SBS_B{args.baseline:.2f}_S{seed_int}.jpg")
                else:
                    # V75.30 Use output_type.py - Left Orig, Right AI
                    sbs_image = create_sbs_image(orig_pil_enh, syn_hq, getattr(args, "output_type", "parallel"))
                    suffix = f"_{args.face_restore.upper()}" if "none" not in args.face_restore else ""
                    if target_long:
                        suffix += f"_{model_type.upper()}{key.upper()}"
                    # V89 FIX: restore Gamma A/B and Clarity suffix
                    if abs(gamma_a_val-1.0) > 0.001:
                        suffix += f"_GA{gamma_a_val:.2f}"
                    if abs(gamma_b_val-1.0) > 0.001:
                        suffix += f"_GB{gamma_b_val:.2f}"
                    clarity_arg = getattr(args, 'clarity', 'none')
                    if clarity_arg and clarity_arg.lower() != "none":
                        c_clean = str(clarity_arg).replace(".", "p").replace("_", "").replace("-", "").upper()
                        suffix += f"_{c_clean}"
                    if abs(args.antialias) > 0.01:
                        suffix += f"_AA{args.antialias:.1f}"
                    if _is_surface_blur_number(args.surface_blur):
                        suffix += f"_BLUR{_get_surface_blur_float(args.surface_blur)}"
                    if args.text_fix > 0:
                        suffix += f"_TEXT{args.text_fix:.1f}"
                    if args.line_fix > 0:
                        suffix += f"_LINE{args.line_fix:.1f}"
                    sbs_path = join(args.output, f"{name}_R{suffix}.3D_SBS_B{args.baseline:.2f}_S{seed_int}.jpg")
            # V75.31 Anti-Alias separated to anti_alias.py
            if abs(args.antialias) > 0.01:
                sbs_image = apply_antialias_to_sbs(sbs_image, args.antialias)
            sbs_image.save(sbs_path, "JPEG", quality=95, subsampling=0)
            print(f"  -> Saved: {basename(sbs_path)}")
        print(f"Processed {start+len(batch_paths)}/{len(image_paths)} images")
def main():
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--config", type=str, default="./configs/stereospace.yaml")
    parser.add_argument("--baseline", type=float, default=0.08)
    parser.add_argument("--mode", type=str, default="orig_right", choices=["orig_right","orig_left","both_generated","symmetric","both"])
    parser.add_argument("--src_intrinsics", type=float, nargs=9)
    parser.add_argument("--tgt_intrinsics", type=float, nargs=9)
    parser.add_argument("--resize_mode", type=str, default="stretch", choices=["letterbox","stretch"])
    parser.add_argument("--high_res", type=int, default=768)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--gamma_a", type=float, default=1.0)
    parser.add_argument("--gamma_b", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--clarity", type=str, default="none")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--antialias", type=float, default=0)
    parser.add_argument("--face_restore", type=str, default="none", choices=["none","gfpgan","gfpgan_half","gfpgan_both"])
    parser.add_argument("--face_fidelity", type=float, default=0.7)
    parser.add_argument("--surface_blur", type=str, default="0")
    parser.add_argument("--clarity_blend", type=str, default="none")
    parser.add_argument("--text_fix", type=float, default=0)
    parser.add_argument("--line_fix", type=float, default=0)
    parser.add_argument("--upscale", type=str, default="none")
    parser.add_argument("--enhance", type=str, default="none")
    parser.add_argument("--output_type", type=str, default="parallel", choices=["parallel","xd","align_parallel","align_xd","align_both"])
    parser.add_argument("--seed", type=int, default=1655838429)
    parser.add_argument("--seam_mode", type=str, default="auto", choices=["auto", "manual", "off"], help="Seam: auto=flattest column in overlap, manual=Seam X %, off=single 768 no tiling")
    parser.add_argument("--seam_x", type=float, default=50.0, help="Manual seam position 0-100% inside overlap. 0=left edge, 100=right edge. Use to push seam off face/hand.")
    args = parser.parse_args()
    # CLAMP: 3D Strength max 0.15 (GUI + CLI cannot exceed)
    try:
        _b = float(args.baseline)
        # GUI 传 0.03..0.15, CLI 传 3..15 也支持: >1 就除以 100
        if _b > 1.0:
            _b = _b / 100.0
        args.baseline = min(max(_b, 0.01), 0.15)
        print(f"[SD3D] 3D Strength {args.baseline:.2f}", flush=True)
    except: pass
    config = OmegaConf.load(args.config)
    # === FIX V80.12: Steps 和 Resolution 强制覆盖 ===
    target_hr = int(getattr(args, "high_res", 768) or 768)
    steps_arg = int(getattr(args, "steps", 8) or 8)

    # Steps
    config.num_inference_steps = steps_arg
    pass  # [SD3D] log suppressed (Easy3D-style)

    # Resolution - 覆盖所有可能限制分辨率的字段
    try:
        config.data.train_width = target_hr
        config.data.train_height = target_hr
        # 旧逻辑里的 target_high_res 字段也同步
        if hasattr(config.data, 'target_high_res'):
            config.data.target_high_res = target_hr
        # yaml顶层的 high_res 字段
        if hasattr(config, 'high_res_long_side'):
            config.high_res_long_side = target_hr
        if hasattr(config, 'max_resolution'):
            config.max_resolution = max(target_hr, int(getattr(config, 'max_resolution', target_hr)))
        if hasattr(config, 'high_res_mode'):
            # 强制改成直接模式，避免 long_side 逻辑把 1152 又压回 1024
            # 如果你想保留 long_side 逻辑，可以注释掉下一行
            pass
        pass  # [SD3D] log suppressed (Easy3D-style)
    except Exception as e:
        pass  # [SD3D] log suppressed (Easy3D-style)
        import traceback; traceback.print_exc()

    generate_novel_view(args, config)
if __name__ == "__main__":
    main()
