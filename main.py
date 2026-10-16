import cv2
import time
from screen_capture import ScreenCapture
from board_detector import BoardDetector, draw_detections
from shot_calculator import ShotCalculator

def main():
    print('=' * 55)
    print('  Carrom Pool Pro AI Assistant (Real-Time Ultra-Fast)')
    print('=' * 55)
    cap = ScreenCapture()
    reg = cap.select_region_interactive()
    if not reg:
        print('[!] No region selected. Exiting.')
        return

    detector = BoardDetector()
    calc = ShotCalculator()

    window_name = 'Carrom Pool AI Assistant (Q to exit)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print('[*] Assistant active!')
    print('[*] Target Mode: STRICTLY BLACK COINS ONLY')
    print('[*] Drag the red aim arrow in Carrom Pool to see the instant laser trajectory.')
    print('[*] Press Q in the window anytime to exit.\n')

    prev_time = time.time()
    fps = 60.0

    while True:
        frame = cap.capture_frame()

        curr_time = time.time()
        instant_fps = 1.0 / max(0.0005, (curr_time - prev_time))
        fps = 0.90 * fps + 0.10 * instant_fps
        prev_time = curr_time

        state = detector.analyze_frame(frame)
        out = draw_detections(frame, state)
        out = calc.calculate_shots(out, state)

        # Status badge & FPS overlay
        status_txt = f"FPS: {int(fps)}"
        cv2.putText(out, status_txt, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        if state.is_actively_aiming:
            cv2.putText(out, "LOCKED AIM", (15, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(out, "READY", (15, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(window_name, out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print('[*] Exited cleanly.')

if __name__ == '__main__':
    main()