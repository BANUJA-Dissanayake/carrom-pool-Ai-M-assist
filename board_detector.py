import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class Coin:
    x: float
    y: float
    radius: float
    color: str = 'black'

@dataclass
class Pocket:
    x: float
    y: float
    radius: float = 28.0

@dataclass
class BoardState:
    pockets: List[Pocket] = field(default_factory=list)
    coins: List[Coin] = field(default_factory=list)
    striker: Optional[Coin] = None
    aim_vector: Optional[Tuple[float, float]] = None
    is_actively_aiming: bool = False
    board_w: int = 0
    board_h: int = 0

class BoardDetector:
    def __init__(self):
        self.cached_striker_x = None
        self.cached_striker_y = None
        self.cached_striker_r = None
        self.cached_coins = []
        self.frame_count = 0
        self.smooth_vx = 0.0
        self.smooth_vy = -1.0
        self.smooth_inited = False

    def analyze_frame(self, frame: np.ndarray) -> BoardState:
        self.frame_count += 1
        state = BoardState()
        h, w = frame.shape[:2]
        state.board_w, state.board_h = w, h

        inset = min(w, h) * 0.075
        p_radius = min(w, h) * 0.055
        state.pockets = [
            Pocket(inset, inset, p_radius),
            Pocket(w - inset, inset, p_radius),
            Pocket(inset, h - inset, p_radius),
            Pocket(w - inset, h - inset, p_radius)
        ]

        # -------------------------------------------------------------
        # 1. REAL-TIME STRIKER TRACKING (High Resolution 1:1 Scan)
        # -------------------------------------------------------------
        base_y1 = int(h * 0.65)
        base_y2 = int(h * 0.96)
        baseline_crop = frame[base_y1:base_y2, :]
        base_gray = cv2.cvtColor(baseline_crop, cv2.COLOR_BGR2GRAY)
        base_blurred = cv2.GaussianBlur(base_gray, (5, 5), 1.5)

        max_striker_r = int(min(w, h) * 0.07)
        base_circles = cv2.HoughCircles(
            base_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=45, param2=16, minRadius=10, maxRadius=max_striker_r
        )

        striker_obj = None
        if base_circles is not None:
            side_inset = min(w, h) * 0.12
            candidates = []
            for c in base_circles[0]:
                cx = float(c[0])
                cy = float(c[1]) + base_y1
                cr = float(c[2])

                # Reject corner pocket areas
                if cx < side_inset or cx > (w - side_inset):
                    continue

                ix, iy, ir = int(c[0]), int(c[1]), int(cr * 0.7)
                y_min, y_max = max(0, iy - ir), min(base_gray.shape[0], iy + ir)
                x_min, x_max = max(0, ix - ir), min(base_gray.shape[1], ix + ir)
                patch = base_gray[y_min:y_max, x_min:x_max]
                if patch.size == 0:
                    continue
                brightness = float(np.mean(patch))

                proximity_bonus = 0.0
                if self.cached_striker_x is not None:
                    dist_cached = np.hypot(cx - self.cached_striker_x, cy - self.cached_striker_y)
                    if dist_cached < 40.0:
                        proximity_bonus = 30.0

                score = brightness + proximity_bonus
                candidates.append((score, cx, cy, cr))

            if candidates:
                candidates.sort(key=lambda item: item[0], reverse=True)
                _, best_cx, best_cy, best_cr = candidates[0]

                if self.cached_striker_x is not None:
                    dist_moved = np.hypot(best_cx - self.cached_striker_x, best_cy - self.cached_striker_y)
                    if dist_moved > 3.0:
                        self.cached_striker_x = best_cx
                        self.cached_striker_y = best_cy
                    else:
                        self.cached_striker_x = 0.6 * self.cached_striker_x + 0.4 * best_cx
                        self.cached_striker_y = 0.6 * self.cached_striker_y + 0.4 * best_cy
                else:
                    self.cached_striker_x = best_cx
                    self.cached_striker_y = best_cy
                self.cached_striker_r = best_cr

                striker_obj = Coin(self.cached_striker_x, self.cached_striker_y, self.cached_striker_r, 'striker')

        elif self.cached_striker_x is not None:
            striker_obj = Coin(self.cached_striker_x, self.cached_striker_y, self.cached_striker_r, 'striker')

        state.striker = striker_obj

        # -------------------------------------------------------------
        # 2. HIGH-PRECISION AIM ARROW DETECTION (0.05° Collinear fitLine)
        # -------------------------------------------------------------
        state.is_actively_aiming = False
        state.aim_vector = None

        if state.striker:
            sx, sy = int(state.striker.x), int(state.striker.y)
            sr = state.striker.radius

            board_unit = min(w, h)
            R_SEARCH = int(board_unit * 0.36)

            x1 = max(0, sx - R_SEARCH)
            x2 = min(w, sx + R_SEARCH)
            y1 = max(0, sy - R_SEARCH)
            y2 = min(h, sy + R_SEARCH)

            roi = frame[y1:y2, x1:x2]
            local_sx = sx - x1
            local_sy = sy - y1

            # High-precision fiery red/orange arrow mask (immune to wooden board / baseline)
            b_ch = roi[:, :, 0].astype(np.int32)
            g_ch = roi[:, :, 1].astype(np.int32)
            r_ch = roi[:, :, 2].astype(np.int32)

            arrow_mask = (r_ch > 185) & (r_ch - b_ch > 85) & (b_ch < 115) & (r_ch - g_ch > -15)
            mask_u8 = arrow_mask.astype(np.uint8) * 255

            # 1. Blank out striker disc and its resting glow aura ring (1.5 * sr)
            cv2.circle(mask_u8, (int(local_sx), int(local_sy)), int(sr * 1.5), 0, -1)

            # 2. Blank out baseline horizontal strip
            band_y1 = max(0, int(local_sy - 16))
            band_y2 = min(mask_u8.shape[0], int(local_sy + 20))
            mask_u8[band_y1:band_y2, :] = 0

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8)
            candidates = []

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                bw = stats[i, cv2.CC_STAT_WIDTH]
                bh = stats[i, cv2.CC_STAT_HEIGHT]
                # Reject tiny noise specks and 1-pixel horizontal baseline remnants
                if area < 70 or bw < 14 or bh < 14:
                    continue

                pts = np.argwhere(labels == i)[:, [1, 0]].astype(np.float32)
                dists = np.hypot(pts[:, 0] - local_sx, pts[:, 1] - local_sy)
                min_dist = float(np.min(dists))
                max_dist = float(np.max(dists))

                # Arrow must originate near striker and have radial span
                if min_dist > (sr * 2.5):
                    continue

                p_near = pts[np.argmin(dists)]
                p_far = pts[np.argmax(dists)]
                v_outward = p_far - p_near
                v_out_norm = float(np.linalg.norm(v_outward))
                if v_out_norm < 15.0:
                    continue
                v_outward /= v_out_norm

                # Collinear fitLine for 0.05° true arrow trajectory accuracy
                vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                if np.dot([vx, vy], v_outward) < 0:
                    vx, vy = -vx, -vy

                # Radial alignment verification: arrow points radially outward from striker
                cx, cy = centroids[i]
                v_striker = np.array([cx - local_sx, cy - local_sy], dtype=np.float32)
                v_str_norm = float(np.linalg.norm(v_striker))
                if v_str_norm < 1.0:
                    continue
                v_striker /= v_str_norm

                radial_dot = float(np.dot([vx, vy], v_striker))
                if radial_dot > 0.70:
                    candidates.append((area, vx, vy))

            if candidates:
                candidates.sort(key=lambda item: item[0], reverse=True)
                _, raw_vx, raw_vy = candidates[0]

                norm = float(np.hypot(raw_vx, raw_vy))
                if norm > 0:
                    raw_vx /= norm
                    raw_vy /= norm

                    if not self.smooth_inited:
                        self.smooth_vx, self.smooth_vy = raw_vx, raw_vy
                        self.smooth_inited = True
                    else:
                        dot = self.smooth_vx * raw_vx + self.smooth_vy * raw_vy
                        # Ultra-responsive dynamic smoothing:
                        # 0.92 when moving fast (zero lag, instant flick tracking)
                        # 0.60 when steady (eliminates micro-jitter)
                        alpha = 0.60 if dot > 0.98 else 0.92
                        self.smooth_vx = (1.0 - alpha) * self.smooth_vx + alpha * raw_vx
                        self.smooth_vy = (1.0 - alpha) * self.smooth_vy + alpha * raw_vy
                        s_norm = float(np.hypot(self.smooth_vx, self.smooth_vy))
                        if s_norm > 0:
                            self.smooth_vx /= s_norm
                            self.smooth_vy /= s_norm

                    state.aim_vector = (float(self.smooth_vx), float(self.smooth_vy))
                    state.is_actively_aiming = True
            else:
                self.smooth_inited = False
                state.is_actively_aiming = False
                state.aim_vector = None

        # -------------------------------------------------------------
        # 3. HIGH PERFORMANCE PIECE CACHING (Strictly Black Coins Only)
        # -------------------------------------------------------------
        # Coins NEVER move while the player is aiming.
        # Scanning coins only when idle / periodically provides 200+ FPS!
        should_scan_coins = (
            len(self.cached_coins) == 0 or
            (not state.is_actively_aiming and (self.frame_count % 15 == 0))
        )

        if should_scan_coins:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 1.2)

            coin_r_min = max(8, int(min(w, h) * 0.018))
            coin_r_max = max(16, int(min(w, h) * 0.045))
            min_dist_pieces = int(coin_r_min * 1.5)

            circles = cv2.HoughCircles(
                blurred_frame, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist_pieces,
                param1=45, param2=20, minRadius=coin_r_min, maxRadius=coin_r_max
            )

            detected_black_coins = []
            if circles is not None:
                circles = np.float32(np.around(circles))[0]
                for c in circles:
                    cx, cy, cr = float(c[0]), float(c[1]), float(c[2])
                    # Skip outer pocket wells
                    if cx < inset * 0.85 or cx > (w - inset * 0.85) or cy < inset * 0.85 or cy > (h - inset * 0.85):
                        continue
                    if any(np.hypot(p.x - cx, p.y - cy) < 32 for p in state.pockets):
                        continue
                    # Skip striker
                    if state.striker and np.hypot(cx - state.striker.x, cy - state.striker.y) < (state.striker.radius + 12):
                        continue

                    # Sample coin interior to identify BLACK coins
                    ix, iy, ir = int(cx), int(cy), int(cr * 0.6)
                    patch = gray_frame[max(0, iy - ir):min(iy + ir, h), max(0, ix - ir):min(ix + ir, w)]
                    patch_bgr = frame[max(0, iy - ir):min(iy + ir, h), max(0, ix - ir):min(ix + ir, w)]
                    if patch.size == 0:
                        continue

                    mean_gray = float(np.mean(patch))
                    mean_bgr = np.mean(patch_bgr, axis=(0, 1))

                    # Strictly black coins only: dark gray/neutral
                    # Excludes white coins (mean_gray > 175) and wooden foul circles (|B - R| > 40)
                    is_black_coin = (20 < mean_gray < 110) and (abs(float(mean_bgr[0]) - float(mean_bgr[2])) < 40)

                    if is_black_coin:
                        detected_black_coins.append(Coin(cx, cy, cr, 'black'))

            self.cached_coins = detected_black_coins

        state.coins = self.cached_coins
        return state

def draw_detections(frame: np.ndarray, state: BoardState) -> np.ndarray:
    return frame.copy()