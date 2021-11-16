#!/usr/bin/env python3
from typing import Tuple

import numpy as np
import imutils
import cv2

import time


class Detector:
    def __init__(self):
        self.height = 700
        self.width = 700
        self.timer_count = 0
        self.frame = None
        self.gray = None
        self.prev_frame = None
        self.left_color = None
        self.right_color = None
        self.time_elapsed = 0
        self.thresholding = False
        self.done_thresholding = False
        self.start_time = 0
        self.color_coords = [(int(self.width * 0.3), int(self.height * 0.3)),
                             (int(self.width * 0.7), int(self.height * 0.3)),
                             (int(self.width * 0.3), int(self.height * 0.7)),
                             (int(self.width * 0.7), int(self.height * 0.7))]
        self.colors = []

    def get_color(self, first: bool):
        """
        Gets the average of the colors inside of the bounding circles and returns the left and right average color
        Args:
            first: Specifies if this is the first or second set of points
    
        Returns:
            The average left color and average right color
        """
        if first:
            for coord in self.color_coords[:2]:
                avg = np.average(self.frame[coord[0] - 2:coord[0] + 2, coord[1] - 2:coord[1] + 2], axis=(0, 1))
                self.colors.append(avg)

    def prompt_color(self) -> np.array:
        """
        Draws the prompt for color selection on the frame and returns it
        Args:
            frame: Input frame
    
        Returns:
    
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame = self.frame.copy()
        org = (int(self.width * 0.228), int(self.height * 0.085714))
        font_scale = 1.0
        txt_color = (255, 255, 255)
        txt_thickness = 2
        cv2.putText(frame, "Press Space When Ready", org, font, font_scale, txt_color, thickness=txt_thickness)

        radius = int(self.width * 0.03)
        color = (255, 255, 255)
        thickness = 2
        for coord in self.color_coords:
            cv2.circle(frame, coord, radius, color, thickness)
            cv2.circle(frame, coord, radius, color, thickness)

        num_coord = (int(self.width * 0.5) - 10, int(self.height * 0.5) - 5)
        if self.thresholding:
            num_scale = 2.0
            cv2.putText(frame, f"{self.timer_count - self.time_elapsed}", num_coord, font, num_scale, txt_color,
                        thickness=txt_thickness)
        return frame

    def impose_blocks(self) -> np.array:
        """
        Draws bounding blocks on the bottom of the screen as different drum components
        Args:

        Returns:
            Output frame
        """

        colors = np.array([[255, 0, 0],
                           [0, 255, 0],
                           [0, 0, 255],
                           [125, 125, 125]])
        height_factor = 3.5
        frame = self.frame.copy()
        for i in range(4):
            new_val = (frame[int((self.height // 4) * height_factor):,
                       (self.width // 4) * i:, :] + colors[i]) / 2
            frame[int((self.height // 4) * height_factor):, (self.width // 4) * i:, :] = new_val
        frame[frame > 255] = 255
        return frame

    def generate_mask(self, gray: np.array) -> np.array:
        delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.bitwise_and(self.frame, self.frame, mask=thresh)

        buffer = 30
        master_mask = np.zeros(thresh.shape[:2], dtype="uint8")
        for color in self.colors:
            mask = cv2.inRange(thresh, (color - buffer).astype(int), (color + buffer).astype(int))
            mask = cv2.dilate(mask, None, iterations=2)
            master_mask = cv2.bitwise_or(mask, master_mask)
        return master_mask

    def manage_thresholding(self):
        self.time_elapsed = int(time.time() - self.start_time)
        if self.time_elapsed >= self.timer_count:
            self.thresholding = False
            self.get_color(True)
            self.done_thresholding = True

    def webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        _, prev_frame = cap.read()
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_frame = cv2.resize(prev_frame, (700, 700))
        self.prev_frame = cv2.flip(prev_frame, 1)

        while True:
            if self.thresholding:
                self.manage_thresholding()

            _, frame = cap.read()
            frame = cv2.resize(frame, (700, 700))
            self.frame = cv2.flip(frame, 1)
            processed = self.impose_blocks()
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            if self.done_thresholding:
                mask = self.generate_mask(gray)
                contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = imutils.grab_contours(contours)
                for c in contours:
                    if cv2.contourArea(c) < 1000:
                        continue
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("Delta", mask)
            else:
                processed = self.prompt_color()

            c = cv2.waitKey(1)
            if c == ord("q"):
                break
            elif c == ord(" ") and not self.done_thresholding and not self.thresholding:
                self.thresholding = True
                self.start_time = time.time()
            self.prev_frame = gray
            cv2.imshow('Input', processed)

        cap.release()
        cv2.destroyAllWindows()


def main():
    detector = Detector()
    detector.webcam()


if __name__ == "__main__":
    main()
