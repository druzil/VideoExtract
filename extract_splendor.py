#!/usr/bin/env python3
"""
Extracts a BGA Splendor move list from a screen-capture video and writes a
CSV identical in structure to the user-supplied sample (only column A filled).

Usage:
    python extract_splendor.py input.mp4 --out moves.csv
"""

import argparse, cv2, numpy as np, pytesseract, pandas as pd, re, os, sys
from collections import Counter, defaultdict
from pathlib import Path

# add this near the top of extract_splendor.py, just after the imports TODO remove
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import itertools, operator
from typing import Tuple, List

HSV = Tuple[int, int, int]

DEFAULT_MIN_AREA: int   = 5_000   # smallest face‑up card area in pixels
DEFAULT_MAX_AREA: int   = 90_000  # largest deck‑back card area in pixels
DEFAULT_MIN_RATIO: float = 1.1    # h/w lower bound (≈1.25 face‑up, ≈1.7 deck)
DEFAULT_MAX_RATIO: float = 1.9

# ---------------------------------------------------------------------------
# 1.  Auto‑derive the blue HSV window
# ---------------------------------------------------------------------------

def auto_blue_range(frame_bgr: np.ndarray,
                    *,
                    sat_min: int =  80,
                    val_min: int =  80,
                    hue_lo:  int =  85,
                    hue_hi:  int = 150,
                    pad_h:   int =  15,
                    pad_sv:  int =  80) -> Tuple[HSV, HSV]:
    """Return (low, high) HSV suitable for cv2.inRange for Splendor card borders.

    Strategy: build a coarse HSV histogram for *pixels that look vaguely blue* –
    saturation & value above `sat_min` / `val_min`, hue in `[hue_lo, hue_hi]`.
    Pick the **modal hue** and widen it by ±`pad_h`; widen S&V by ±`pad_sv`.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    mask = (S > sat_min) & (V > val_min) & (H >= hue_lo) & (H <= hue_hi)
    if not np.any(mask):                    # fallback: entire blue window
        low, high = (hue_lo, sat_min, val_min), (hue_hi, 255, 255)
        return low, high

    # histogram of H values (0‑179)
    hist = np.bincount(H[mask].flatten(), minlength=180)
    peak = int(np.argmax(hist))

    low  = (max(peak - pad_h, 0),              max(int(S[mask].min()) - pad_sv, 0),
            max(int(V[mask].min()) - pad_sv, 0))
    high = (min(peak + pad_h, 179),            min(int(S[mask].max()) + pad_sv, 255),
            min(int(V[mask].max()) + pad_sv, 255))
    return low, high

# ---------------------------------------------------------------------------
# 2.  Slot detection
# ---------------------------------------------------------------------------

def detect_card_slots(frame_bgr: np.ndarray,
                       low: HSV | None = None,
                       high: HSV | None = None,
                       *,
                       min_area: int = DEFAULT_MIN_AREA,
                       max_area: int = DEFAULT_MAX_AREA,
                       min_ratio: float = DEFAULT_MIN_RATIO,
                       max_ratio: float = DEFAULT_MAX_RATIO) -> List[Tuple[int, int]]:
    """Return [(y, x), …] for the 12 face‑up card slots.

    If *low/high* are *None* they’re auto‑derived from the frame.
    Raises *RuntimeError* if 12 rectangles aren’t found.
    """
    if low is None or high is None:
        low, high = auto_blue_range(frame_bgr)

    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low, high)
    mask = cv2.dilate(mask, None, iterations=3)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area  = w * h
        if not (min_area <= area <= max_area):
            continue
        ratio = h / w
        if not (min_ratio <= ratio <= max_ratio):
            continue
        rects.append((x, y, w, h))

    # keep the 12 left‑most rectangles (filters out right‑hand UI blue)
    rects = sorted(rects, key=lambda r: r[0])[:12]
    rects.sort(key=lambda r: (r[1], r[0]))  # row‑major order

    if len(rects) != 12:
        raise RuntimeError(
            f"Expected 12 card slots, found {len(rects)} – adjust thresholds or crop.")

    return [(y, x) for x, y, _w, _h in rects]

# ---------------------------------------------------------------------------
# 3.  Debug helper – write mask & rectangles to PNGs
# ---------------------------------------------------------------------------

def debug_blue_mask(frame_bgr: np.ndarray,
                    low: HSV | None = None,
                    high: HSV | None = None,
                    *,
                    min_area: int = DEFAULT_MIN_AREA,
                    max_area: int = DEFAULT_MAX_AREA,
                    min_ratio: float = DEFAULT_MIN_RATIO,
                    max_ratio: float = DEFAULT_MAX_RATIO,
                    save_prefix: str = "debug") -> None:
    if low is None or high is None:
        low, high = auto_blue_range(frame_bgr)

    hsv   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv, low, high)
    cv2.imwrite(f"{save_prefix}_mask.png", mask)

    vis   = frame_bgr.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area  = w * h
        ratio = h / w
        if not (min_area <= area <= max_area):
            continue
        if not (min_ratio <= ratio <= max_ratio):
            continue
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite(f"{save_prefix}_boxes.png", vis)

    print(f"[debug] wrote {save_prefix}_mask.png and {save_prefix}_boxes.png")


#### ---------- helper: colour → initial ----------
COL2INIT = {"white":"W", "blue":"B", "green":"G", "red":"R", "black":"L"}  # ‘Onyx’→L

def nearest_gem_colour(bgr):
    """Return Splendor gem name from a BGR pixel sample (top-stripe)."""
    # reference BGR means taken from typical board screenshots
    ref = {"white":(245,245,245), "blue":(200,110,20),
           "green":(60,160,60),  "red":(60,60,180),  "black":(30,30,30)}
    b,g,r = bgr
    def dist(c): B,G,R = ref[c]; return (B-b)**2+(G-g)**2+(R-r)**2
    return min(ref, key=dist)

#### ---------- OCR utilities ----------
tess_cfg_line = r"--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-,()"

re_get   = re.compile(r"gets? (\d) tokens?:? ?([A-Za-z, ]+)")
re_buy   = re.compile(r"buys? .*row *(\d)")
re_res   = re.compile(r"reserves? .*row *(\d)")
re_give  = re.compile(r"gives? back")

#### ---------- main routine ----------
def main(video: Path, out: Path):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened(): sys.exit("Cannot open video.")
    fps  = cap.get(cv2.CAP_PROP_FPS)
    H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # 1 . find banner strip once – assume it’s the first wide white bar ≥80% white
    ret, first = cap.read()

    probe_x, probe_y = 200,  160        # tweak if that misses the blue outline
    bgr = first[probe_y, probe_x]       # (B,G,R)
    h,s,v = cv2.cvtColor(
                np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    print(f"[HSV probe] border pixel at ({probe_x},{probe_y}) → H={h}, S={s}, V={v}")

    low  = (max(h-15, 0),   max(s-80, 0),    max(v-80, 0))
    high = (min(h+15, 179), min(s+80, 255),  min(v+80, 255))
    print("[debug] blue HSV range", low, high)


    # ---------- card-slot debug image ----------------------------------------
    LOW_HSV, HIGH_HSV = auto_blue_range(first)
    debug_blue_mask(first, "slot_finder")   # will create three PNGs
    slots = detect_card_slots(first, LOW_HSV, HIGH_HSV)
    row_h = int(np.median([slots[4+i][0] - slots[i][0] for i in range(4)]))   # vertical step
    col_w = int(np.median([slots[i+1][1] - slots[i][1] for i in (0,1,2,4,5,6,8,9,10)]))  # horiz step

    debug_img = first.copy()
    for (y, x) in slots:
        cv2.circle(debug_img, (x, y), 8, (0, 0, 255), -1)   # red filled dot

    cv2.imwrite("card_slots_debug.png", debug_img)
    print("[debug] wrote card_slots_debug.png – verify red dots match slot corners")
    # --------------------------------------------------------------------------


    gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    white_mask = (gray > 230).astype(np.uint8)

    runs = []
    start = None
    for y in range(gray.shape[0]):
        if white_mask[y].mean() > 0.75:          # row is mostly white / light grey
            if start is None:
                start = y
        else:
            if start is not None:
                runs.append((start, y - 1))
                start = None
    if start is not None:                        # catch run that reaches bottom
        runs.append((start, gray.shape[0] - 1))

    # choose the widest white band (height in pixels)
    banner_top, banner_bottom = max(runs, key=lambda r: r[1] - r[0])
    banner_y = banner_top
    banner_h = banner_bottom - banner_top + 1                                                        

    def crop_banner(f): return f[banner_y:banner_y+banner_h, :]

    moves = []                     # compact move strings for column A
    last_text = ""

    #### pass 1 – detect banner-change frames #################################################
    step = int(fps*0.5)            # every ½ s
    frames_of_interest = []
    idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read();  idx += 1

        # inside while True (the ½-second scan) TODO remove
        if idx % step == 0:
            sample_txt = pytesseract.image_to_string(crop_banner(frame), config=tess_cfg_line).strip()
            if idx < 20:           # only print the first few so your console doesn’t explode
                print(idx, repr(sample_txt))
        
        if not ret: break
        if idx % step: continue
        text = pytesseract.image_to_string(crop_banner(frame), config=tess_cfg_line).strip()
        if text and text != last_text and any(k in text.lower() for k in
                                              ("gets","buys","reserves","gives")):
            frames_of_interest.append(idx)
            last_text = text
            print(f"Frames of interest {idx}")

        # TODO remove
        if idx > 500:
            break    

    #### pass 2 – for each candidate, scoot ±0.3 s to find the true end-of-move frame ########
    for fid in frames_of_interest:
        best_frame, best_text = None, ""
        for off in range(-int(0.3*fps), int(0.3*fps)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid+off)
            ret, fr = cap.read()
            if not ret: continue
            txt = pytesseract.image_to_string(crop_banner(fr), config=tess_cfg_line).strip()
            if txt and any(k in txt.lower() for k in
                           ("gets","buys","reserves","gives")):
                best_frame, best_text = fr, txt
        if best_frame is None: continue    # junk snapshot

        low = best_text.lower()

        print(f"test to analyse {low}")
        # ---------- TAKE GEMS ----------
        if "gets" in low:
            print(f"found gets")
            m = re_get.search(best_text)
            if not m: continue
            n_tokens = m.group(1)
            colours  = [COL2INIT.get(c.strip().lower()[0]+"hite" if c.lower().startswith("diam") else c.strip().lower(), "?")
                        for c in m.group(2).split(",")]
            cost = Counter(colours)
            cost_code = "".join(f"{v}{k}" for k,v in sorted(cost.items()))
            moves.append(f'"{n_tokens} - {cost_code}"')
            continue

        # ---------- RESERVE / BUY ----------
        is_reserve = "reserve" in low
        print(f"is reserve {is_reserve}")
        row_match  = re_res.search(best_text) if is_reserve else re_buy.search(best_text)
        print(f"row_match {row_match}")
        if not row_match: continue
        row = int(row_match.group(1))                       # 0,1,2

        # Extract 4-column grid coords once (works because zoom is constant).
        # each slot ~15 % width, centred horizontally; empirically:
        col_w = int(0.15*W);  x0 = int(0.27*W)              # tune if needed
        row_h = int(0.20*H);  y0 = int(0.28*H)
        slots = [(y0+row_h*r, x0+col_w*c) for r in range(3) for c in range(4)]

        # Grab the slot bounding box just *before* the change (2 frames earlier)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid-2)
        ret, before = cap.read();        assert ret
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid+2)
        ret, after  = cap.read();        assert ret

        # find which slot changed >10 % pixels
        changed = None
        for r in range(3):
            for c in range(4):
                y,x = slots[r*4+c]
                card_before = before[y:y+row_h, x:x+col_w]
                card_after  = after[y:y+row_h,  x:x+col_w]
                diff = np.mean(cv2.absdiff(card_before, card_after))
                if diff > 20: changed = (r,c,card_before); break
            if changed: break
        if not changed: continue
        r,c,card_img = changed
        print(f"found changed {r} {c} {card_img}")
        # (a) produced colour  – average strip 12 px at top-centre
        stripe = card_img[5:17, col_w//2-20:col_w//2+20]
        prod   = nearest_gem_colour(np.mean(stripe.reshape(-1,3),axis=0).astype(int))
        prod_letter = COL2INIT[prod]

        # (b) points – OCR the central big digit
        pts = pytesseract.image_to_string(card_img[row_h//4:row_h//2,
                                                   col_w//2-40:col_w//2+40],
                                          config="--psm 7 -c tessedit_char_whitelist=0123456789").strip()
        pts = pts if pts else "0"

        # (c) cost code – HSV mask for each gem colour icon, count blobs
        hsv   = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
        def count(col):
            # crude HSV ranges; tune if needed
            ranges = {"white":((0,0,220),(180,30,255)),
                      "blue": ((100, 90, 50),(130,255,255)),
                      "green":((50,50,50),(85,255,255)),
                      "red1": ((0,50,50),(10,255,255)),
                      "red2": ((160,50,50),(180,255,255)),
                      "black":((0,0,0),(180,255,40))}
            if col=="red":
                mask = cv2.inRange(hsv, *ranges["red1"])|cv2.inRange(hsv,*ranges["red2"])
            else:
                mask = cv2.inRange(hsv, *ranges[col])
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # each gem icon is ~300 px, ignore tiny noise
            return sum(1 for c in cnts if cv2.contourArea(c) > 200)

        cost_counts = {k:count(k) for k in ("white","blue","green","red","black")}
        cost_code   = "".join(f"{v}{COL2INIT[k]}" for k,v in cost_counts.items() if v)

        move_type = "Reserve" if is_reserve else "Buy"
        moves.append(f'"{move_type} - {prod_letter} - {pts} - {cost_code}"')

        # ---------- EARLY-EXIT SNIPPET ----------
        if len(moves) >= 5:
            break        # leaves the fid-loop
        # ----------------------------------------

    #### -------- write CSV ---------------------------------------------------------------
    header = ["Player", "Selected Move", "Move Eval",
              "Best/Median Move", "Move Eval", "Difference"]

    df = pd.DataFrame(moves, columns=[header[0]])   # put moves in column A
    df = df.reindex(columns=header)                 # add five empty columns

    df.to_csv(out, index=False)
    print(f"Wrote {len(moves)} moves → {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--out", type=Path, default=Path("splendor_moves.csv"))
    main(**vars(p.parse_args()))
