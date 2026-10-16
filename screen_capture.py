import cv2
import numpy as np
import mss

class ScreenCapture:
    def __init__(self, region=None):
        self.sct = mss.mss()
        self.region = region

    def select_region_interactive(self):
        """
        Interactive custom board selection like Version 1:
        Grabs screen, lets user drag a rectangle around ONLY the Carrom board,
        and press ENTER or SPACE.
        """
        print('\n' + '='*55)
        print('[*] CUSTOM BOARD AREA SELECTION')
        print('[*] 1. A screenshot window will open.')
        print('[*] 2. Drag a box over ONLY the Carrom Board.')
        print('[*] 3. Press ENTER or SPACE to confirm (or C to cancel).')
        print('='*55 + '\n')

        # Grab primary monitor
        monitor = self.sct.monitors[1]
        raw = np.asarray(self.sct.grab(monitor))[:, :, :3]

        h, w = raw.shape[:2]
        scale = min(1.0, 1920.0 / max(1, w))
        disp = cv2.resize(raw, (int(w * scale), int(h * scale)))

        window_name = 'Select Carrom Board Only (Drag box -> Press ENTER)'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        roi = cv2.selectROI(window_name, disp, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()

        # If user pressed cancel or selected tiny area
        if roi[2] < 20 or roi[3] < 20:
            print('[!] Selection cancelled or invalid.')
            return None

        self.region = {
            'left': int(roi[0] / scale) + monitor['left'],
            'top': int(roi[1] / scale) + monitor['top'],
            'width': int(roi[2] / scale),
            'height': int(roi[3] / scale)
        }
        print(f"[+] Custom board selected: {self.region}\n")
        return self.region

    def capture_frame(self):
        return np.asarray(self.sct.grab(self.region))[:, :, :3]