import math
import cv2
import numpy as np

class ShotCalculator:
    @classmethod
    def draw_laser_line(cls, img: np.ndarray, pt1, pt2, core_color=(255, 255, 255), glow_color=(0, 215, 255), is_main=False):
        """Draws a smooth Lulubox/BitAIM laser line with outer neon aura glow and crisp solid white core."""
        p1 = (int(pt1[0]), int(pt1[1]))
        p2 = (int(pt2[0]), int(pt2[1]))
        glow_thick = 7 if is_main else 5
        core_thick = 3 if is_main else 2
        # Outer neon glow
        cv2.line(img, p1, p2, glow_color, glow_thick, cv2.LINE_AA)
        # Inner white laser core
        cv2.line(img, p1, p2, core_color, core_thick, cv2.LINE_AA)

    @classmethod
    def calculate_shots(cls, frame: np.ndarray, state) -> np.ndarray:
        output = frame.copy()

        # 1. CLEAN IDLE STATE: Zero lines when resting or sliding disc without dragging arrow
        if not state.is_actively_aiming or not state.striker or state.aim_vector is None:
            return output

        h, w = frame.shape[:2]
        inset = min(w, h) * 0.076
        wall_left, wall_right = inset, w - inset
        wall_top, wall_bottom = inset, h - inset

        s_radius = float(state.striker.radius)
        c_radius = float(s_radius * 0.74)
        contact_thresh = s_radius + c_radius

        s_pos = np.array([state.striker.x, state.striker.y], dtype=np.float32)
        dir_aim = np.array(state.aim_vector, dtype=np.float32)

        # -------------------------------------------------------------
        # 2. RAYCASTING STRIKER AIM PATH (Up to 3 cushion bounces)
        # -------------------------------------------------------------
        curr_pos = s_pos.copy()
        curr_dir = dir_aim.copy()
        striker_path = [curr_pos.copy()]

        hit_coin = None
        ghost_ball_pos = None

        for bounce in range(3):
            closest_dist = float('inf')
            closest_c = None
            closest_gb = None

            if state.coins:
                for coin in state.coins:
                    c_pos = np.array([coin.x, coin.y], dtype=np.float32)
                    vec = c_pos - curr_pos
                    proj = float(np.dot(vec, curr_dir))
                    if proj <= 0.5:
                        continue

                    perp = float(np.linalg.norm(vec - proj * curr_dir))
                    if perp < contact_thresh:
                        offset = math.sqrt(max(0.0, contact_thresh**2 - perp**2))
                        dist = proj - offset
                        if 0.5 < dist < closest_dist:
                            closest_dist = dist
                            closest_c = coin
                            closest_gb = curr_pos + curr_dir * dist

            t_wall, wall_norm = cls.find_wall_intersection(
                curr_pos, curr_dir, wall_left, wall_right, wall_top, wall_bottom, s_radius
            )

            # Hit coin before reaching cushion wall
            if closest_c is not None and closest_dist < t_wall:
                hit_coin = closest_c
                ghost_ball_pos = closest_gb
                striker_path.append(closest_gb.copy())
                break

            # Cushion bounce
            bounce_point = curr_pos + curr_dir * t_wall
            striker_path.append(bounce_point.copy())
            curr_pos = bounce_point
            curr_dir = curr_dir - 2.0 * float(np.dot(curr_dir, wall_norm)) * wall_norm

        # -------------------------------------------------------------
        # 3. DRAW PRIMARY STRIKER AIM LINE (Laser Neon Gold/Yellow)
        # -------------------------------------------------------------
        for i in range(len(striker_path) - 1):
            cls.draw_laser_line(output, striker_path[i], striker_path[i + 1],
                                core_color=(255, 255, 255), glow_color=(0, 215, 255), is_main=True)

        # -------------------------------------------------------------
        # 4. DRAW TARGET COIN TRAJECTORY & DEFLECTION UPON CONTACT
        # -------------------------------------------------------------
        if hit_coin is not None and ghost_ball_pos is not None:
            c_pos = np.array([hit_coin.x, hit_coin.y], dtype=np.float32)

            col_normal = c_pos - ghost_ball_pos
            norm_len = float(np.linalg.norm(col_normal))

            if norm_len > 0:
                coin_dir = col_normal / norm_len

                # Path of hit coin (with cushion reflections into pockets)
                coin_path, potted_pocket = cls.trace_coin_path(
                    c_pos, coin_dir, wall_left, wall_right, wall_top, wall_bottom, state.pockets, c_radius
                )

                # Bright Neon Green if heading directly into a pocket! Neon Cyan otherwise
                coin_glow = (0, 255, 0) if potted_pocket is not None else (255, 200, 0)

                for i in range(len(coin_path) - 1):
                    cls.draw_laser_line(output, coin_path[i], coin_path[i + 1],
                                        core_color=(255, 255, 255), glow_color=coin_glow, is_main=True)

                # Striker 90-degree deflection trajectory
                unit_tangent = np.array([-coin_dir[1], coin_dir[0]], dtype=np.float32)
                deflect_dir = unit_tangent if np.dot(curr_dir, unit_tangent) > 0 else -unit_tangent
                striker_deflect_path = cls.trace_extended_ray(
                    ghost_ball_pos, deflect_dir, wall_left, wall_right, wall_top, wall_bottom, s_radius, length=500.0
                )

                for i in range(len(striker_deflect_path) - 1):
                    cls.draw_laser_line(output, striker_deflect_path[i], striker_deflect_path[i + 1],
                                        core_color=(255, 255, 255), glow_color=(0, 140, 255), is_main=False)

                # Ghost Ball contact ring with target center dot
                gb_pt = (int(ghost_ball_pos[0]), int(ghost_ball_pos[1]))
                cv2.circle(output, gb_pt, int(s_radius), (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(output, gb_pt, 4, (0, 255, 0), -1, cv2.LINE_AA)

                # Potted feedback: Green glowing target ring and solid white potted puck
                if potted_pocket is not None:
                    p_pt = (int(potted_pocket[0]), int(potted_pocket[1]))
                    cv2.circle(output, p_pt, int(s_radius * 0.9), (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(output, p_pt, int(s_radius * 1.35), (0, 255, 0), 3, cv2.LINE_AA)

        return output

    @staticmethod
    def find_wall_intersection(pos, direction, left, right, top, bottom, radius=0.0):
        # Accurate cushion boundary accounting for ball radius
        l = left + radius
        r = right - radius
        t = top + radius
        b = bottom - radius

        candidates, normals = [], []
        if direction[0] < -1e-5:
            candidates.append((l - pos[0]) / direction[0])
            normals.append(np.array([1, 0], dtype=np.float32))
        elif direction[0] > 1e-5:
            candidates.append((r - pos[0]) / direction[0])
            normals.append(np.array([-1, 0], dtype=np.float32))

        if direction[1] < -1e-5:
            candidates.append((t - pos[1]) / direction[1])
            normals.append(np.array([0, 1], dtype=np.float32))
        elif direction[1] > 1e-5:
            candidates.append((b - pos[1]) / direction[1])
            normals.append(np.array([0, -1], dtype=np.float32))

        valid = [(dist, norm) for dist, norm in zip(candidates, normals) if dist > 0.5]
        if not valid:
            return 800.0, np.array([0, 1], dtype=np.float32)
        return min(valid, key=lambda x: x[0])

    @classmethod
    def trace_coin_path(cls, start_pos, direction, left, right, top, bottom, pockets, radius, max_bounces=2):
        path = [start_pos.copy()]
        curr_pos = start_pos.copy()
        curr_dir = direction.copy()
        potted_pocket = None

        for _ in range(max_bounces + 1):
            t_wall, wall_norm = cls.find_wall_intersection(curr_pos, curr_dir, left, right, top, bottom, radius)

            hit_pocket = None
            min_p_dist = float('inf')
            for p in pockets:
                p_pos = np.array([p.x, p.y], dtype=np.float32)
                vec = p_pos - curr_pos
                proj = float(np.dot(vec, curr_dir))
                if 0.5 < proj < t_wall:
                    perp = float(np.linalg.norm(vec - proj * curr_dir))
                    if perp < p.radius and proj < min_p_dist:
                        min_p_dist = proj
                        hit_pocket = p_pos

            if hit_pocket is not None:
                path.append(hit_pocket)
                potted_pocket = hit_pocket
                break

            bounce_pt = curr_pos + curr_dir * t_wall
            path.append(bounce_pt.copy())
            curr_pos = bounce_pt
            curr_dir = curr_dir - 2.0 * float(np.dot(curr_dir, wall_norm)) * wall_norm

        return path, potted_pocket

    @classmethod
    def trace_extended_ray(cls, start_pos, direction, left, right, top, bottom, radius, length=500.0):
        t_wall, wall_norm = cls.find_wall_intersection(start_pos, direction, left, right, top, bottom, radius)
        if length < t_wall:
            return [start_pos.copy(), start_pos + direction * length]
        bounce_pt = start_pos + direction * t_wall
        rem = length - t_wall
        new_dir = direction - 2.0 * float(np.dot(direction, wall_norm)) * wall_norm
        end_pt = bounce_pt + new_dir * rem
        return [start_pos.copy(), bounce_pt, end_pt]