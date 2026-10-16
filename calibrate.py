import cv2
import numpy as np
import mss

def nothing(x):
    pass

def main():
    sct = mss.mss()
    mon = sct.monitors[1]
    shot = np.array(sct.grab(mon))
    shot = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
    cv2.namedWindow('Calibration')
    cv2.createTrackbar('H Low', 'Calibration', 0, 180, nothing)
    cv2.createTrackbar('S Low', 'Calibration', 0, 255, nothing)
    cv2.createTrackbar('V Low', 'Calibration', 0, 255, nothing)
    cv2.createTrackbar('H High', 'Calibration', 180, 180, nothing)
    cv2.createTrackbar('S High', 'Calibration', 255, 255, nothing)
    cv2.createTrackbar('V High', 'Calibration', 255, 255, nothing)
    while True:
        hsv = cv2.cvtColor(shot, cv2.COLOR_BGR2HSV)
        l = np.array([cv2.getTrackbarPos('H Low', 'Calibration'), cv2.getTrackbarPos('S Low', 'Calibration'), cv2.getTrackbarPos('V Low', 'Calibration')])
        u = np.array([cv2.getTrackbarPos('H High', 'Calibration'), cv2.getTrackbarPos('S High', 'Calibration'), cv2.getTrackbarPos('V High', 'Calibration')])
        mask = cv2.inRange(hsv, l, u)
        res = cv2.bitwise_and(shot, shot, mask=mask)
        cv2.imshow('Mask Preview (Q to quit)', cv2.resize(res, (640, 360)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()