#!/usr/bin/env python3
from typing import Tuple

import numpy as np
import cv2

import time


def get_color(frame: np.array) -> Tuple[np.array, np.array]:
    """
    Gets the average of the colors inside of the bounding circles and returns the left and right average color
    Args:
        frame: Input frame

    Returns:
        The average left color and average right color
    """
    height, width, _ = frame.shape
    left_color = np.average(
        frame[int(width * 0.3) - 2:int(width * 0.3) + 2, int(height * 0.3) - 2:int(height * 0.3) + 2], axis=(0, 1))
    right_color = np.average(
        frame[int(width * 0.7) - 2:int(width * 0.7) + 2, int(height * 0.3) - 2:int(height * 0.3) + 2], axis=(0, 1))
    return left_color, right_color


def prompt_color(frame: np.array, counting: bool = False, seconds_left: int = 0) -> np.array:
    """
    Draws the prompt for color selection on the frame and returns it
    Args:
        frame: Input frame
        counting: Determines whether or not a countdown is occurring
        seconds_left: Elapsed time in the countdown

    Returns:

    """
    height, width, _ = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (int(width * 0.228), int(height * 0.085714))
    font_scale = 1.0
    txt_color = (255, 255, 255)
    txt_thickness = 2
    cv2.putText(frame, "Press Space When Ready", org, font, font_scale, txt_color, thickness=txt_thickness)

    left_coord = (int(width * 0.3), int(height * 0.3))
    right_coord = (int(width * 0.7), int(height * 0.3))
    radius = int(width * 0.03)
    color = (255, 255, 255)
    thickness = 2
    cv2.circle(frame, left_coord, radius, color, thickness)
    cv2.circle(frame, right_coord, radius, color, thickness)

    num_coord = (int(width * 0.5) - 10, int(height * 0.5) - 5)
    if counting:
        num_scale = 2.0
        cv2.putText(frame, f"{3 - seconds_left}", num_coord, font, num_scale, txt_color, thickness=txt_thickness)
    return frame


def impose_blocks(frame: np.array) -> np.array:
    """
    Draws bounding blocks on the bottom of the screen as different drum components
    Args:
        frame: Input frame

    Returns:
        Output frame
    """
    height, width, _ = frame.shape

    colors = np.array([[255, 0, 0],
                       [0, 255, 0],
                       [0, 0, 255],
                       [125, 125, 125]])
    height_factor = 3.5
    frame = frame.copy()
    for i in range(4):
        new_val = (frame[int((height // 4) * height_factor):,
                   (width // 4) * i:, :] + colors[i]) / 2
        frame[int((height // 4) * height_factor):, (width // 4) * i:, :] = new_val
    frame[frame > 255] = 255
    return frame


def webcam():
    thresholding = False
    done_thresholding = False
    start_time = None
    time_elapsed = 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    _, prev_frame = cap.read()
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_frame = cv2.resize(prev_frame, (700, 700))
    prev_frame = cv2.flip(prev_frame, 1)
    while True:
        if thresholding:
            time_elapsed = int(time.time() - start_time)
            if time_elapsed >= 4:
                thresholding = False
                left_color, right_color = get_color(frame)
                done_thresholding = True
        ret, frame = cap.read()
        frame = cv2.resize(frame, (700, 700))
        frame = cv2.flip(frame, 1)
        processed = impose_blocks(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if done_thresholding:
            delta = cv2.absdiff(prev_frame, gray)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            thresh = cv2.bitwise_and(frame, frame, mask=thresh)
            buffer = 100
            mask_left = cv2.inRange(thresh, (left_color - buffer).astype(int), (left_color + buffer).astype(int))
            mask_right = cv2.inRange(thresh, (right_color - buffer).astype(int), (left_color + buffer).astype(int))
            mask_left = cv2.dilate(mask_left, None, iterations=4)

            cv2.imshow("Delta", mask_left)
        else:
            processed = prompt_color(frame, thresholding, time_elapsed)

        cv2.imshow('Input', processed)

        c = cv2.waitKey(1)
        if c == ord("q"):
            break
        elif c == ord(" ") and not done_thresholding and not thresholding:
            thresholding = True
            start_time = time.time()
        prev_frame = gray
    cap.release()
    cv2.destroyAllWindows()


def main():
    webcam()


if __name__ == "__main__":
    main()
