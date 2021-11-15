#!/usr/bin/env python3
import numpy as np
import cv2


def impose_blocks(frame: np.array) -> np.array:
    height, width, _ = frame.shape

    colors = np.array([[255, 0, 0],
                       [0, 255, 0],
                       [0, 0, 255],
                       [125, 125, 125]])
    height_factor = 2.6
    for i in range(4):
        frame[int((height // 4) * height_factor):, (width // 4) * i:, :] = (frame[int((height // 4) * height_factor):,
                                                                            (width // 4) * i:, :] + colors[i]) / 2
    frame[frame > 255] = 255
    return frame


def webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        frame = cv2.flip(frame, 1)
        frame = impose_blocks(frame)
        cv2.imshow('Input', frame)
        c = cv2.waitKey(1)
        if c == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    webcam()


if __name__ == "__main__":
    main()
