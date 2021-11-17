#!/usr/bin/env python3
from typing import Tuple, List

import numpy as np
import imutils
import cv2
import playsound

import time
from enum import Enum
import pickle
import argparse


class States(Enum):
    INIT = 0
    COUNTDOWN_1 = 1
    DONE_1 = 2
    COUNTDOWN_2 = 3
    DONE_2 = 4
    CONFIRM = 5


class CollisionBox:
    def __init__(self, name: str, song_file: str, color: np.array):
        self.last_collision = 0
        self.name = name
        self.song_file = song_file
        self.color = np.array(color)

    def has_collision(self) -> None:
        """
        Checks if enough time has passed to play a sound

        Returns:
            None
        """
        if float(time.time() - self.last_collision) > 0.5:
            print(self.name)
            self.last_collision = time.time()
            playsound.playsound(self.song_file, False)

    def reset_collision(self) -> None:
        self.last_collision = time.time() - 1


class Detector:
    def __init__(self, prev: bool):
        self.height = 700
        self.width = 700
        self.timer_count = 2
        self.frame = None
        self.gray = None
        self.frame_queue = []
        self.time_elapsed = 0
        self.start_time = 0
        self.colors = []
        if not prev:
            self.state = States.INIT
        else:
            self.state = States.CONFIRM
            with open("colors.pickle", "rb") as f:
                self.colors = pickle.load(f)
        self.circle_coordinates = [(int(self.width * 0.3), int(self.height * 0.3)),
                                   (int(self.width * 0.7), int(self.height * 0.3)),
                                   (int(self.width * 0.3), int(self.height * 0.7)),
                                   (int(self.width * 0.7), int(self.height * 0.7))]

        self.height_thresh = self.height * 0.8

        self.boxes = [CollisionBox("one", "drum1.mp3", (255, 0, 0)), CollisionBox("two", "drum2.mp3", (0, 255, 0)),
                      CollisionBox("three", "drum3.mp3", (0, 0, 255)), CollisionBox("four", "drum4.mp3", (255, 0, 255))]
        self.webcam()

    def get_color(self) -> None:
        """
        gets the average of the colors within the bounding circles respectively

        Returns:
            None
        """
        if self.state == States.DONE_1:
            coords = self.circle_coordinates[:2]
        elif self.state == States.DONE_2:
            coords = self.circle_coordinates[2:]
        else:
            coords = []
        for coord in coords:
            avg = np.average(self.frame[coord[1] - 2:coord[1] + 2, coord[0] - 2:coord[0] + 2], axis=(0, 1))
            self.colors.append(avg)

    def prompt_color(self) -> np.array:
        """
        Draws the prompt for color selection on the frame and returns it
    
        Returns:
            Modified frame with prompt
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame = self.frame.copy()
        origin = (int(self.width * 0.228), int(self.height * 0.085714))
        font_scale = 1.0
        txt_color = (255, 255, 255)
        txt_thickness = 2
        radius = int(self.width * 0.03)
        color = (255, 255, 255)
        thickness = 2
        # Displays color fill circles and prompts user to calibrate
        if self.state in (States.INIT, States.COUNTDOWN_1, States.DONE_1, States.COUNTDOWN_2):
            cv2.putText(frame, "Press Space When Ready", origin, font, font_scale, (0, 0, 0),
                        thickness=txt_thickness * 2)
            cv2.putText(frame, "Press Space When Ready", origin, font, font_scale, txt_color, thickness=txt_thickness)

            for coord in self.circle_coordinates:
                cv2.circle(frame, coord, radius, color, thickness)
                cv2.circle(frame, coord, radius, color, thickness)
            if self.state in (States.DONE_1, States.COUNTDOWN_2, States.DONE_2):
                for track_color, coord in zip(self.colors, self.circle_coordinates):
                    cv2.circle(frame, tuple(coord), radius - 1, track_color, -1)
            num_coord = (int(self.width * 0.5) - 10, int(self.height * 0.5) - 5)
            if self.state in (States.COUNTDOWN_1, States.COUNTDOWN_2):
                num_scale = 2.0
                cv2.putText(frame, f"{self.timer_count - self.time_elapsed}", num_coord, font, num_scale, txt_color,
                            thickness=txt_thickness)
        # Prompts user to confirm the colors
        elif self.state == States.DONE_2:
            prompt = "Press Space to Confirm, Esc to Redo"
            origin = (int(self.width * 0.19), int(self.height * 0.085714))
            cv2.putText(frame, prompt, origin, font, 0.8, (0, 0, 0), thickness=4)
            cv2.putText(frame, prompt, origin, font, 0.8, txt_color, thickness=2)
            for coord in self.circle_coordinates:
                cv2.circle(frame, coord, radius, color, thickness)
                cv2.circle(frame, coord, radius, color, thickness)
            if self.state in (States.DONE_1, States.COUNTDOWN_2, States.DONE_2):
                for track_color, coord in zip(self.colors, self.circle_coordinates):
                    cv2.circle(frame, tuple(coord), radius - 1, track_color, -1)
        return frame

    def impose_blocks(self) -> np.array:
        """
        Draws bounding blocks on the bottom of the screen as different drum components

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
            scale_factor = 1.0 if float(time.time() - self.boxes[i].last_collision) < 0.2 else 0.7
            new_val = (frame[int((self.height // 4) * height_factor):,
                       (self.width // 4) * i:(self.width // 4) * (i + 1), :]
                       + self.boxes[i].color * scale_factor) / 2
            frame[int((self.height // 4) * height_factor):, (self.width // 4) * i:(self.width // 4) * (i + 1),
            :] = new_val
        frame[frame > 255] = 255
        return frame

    def generate_mask(self, gray: np.array) -> np.array:
        """
        Uses motion estimation then color thresholding to create a mask to be used for contour detection
        Args:
            gray: Grayscale image used for calculating the motion mask

        Returns:
            Output mask
        """
        if self.frame_queue:
            # Motion estimation
            delta = cv2.absdiff(self.frame_queue[0], gray)
            thresh = cv2.threshold(delta, 20, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, np.ones((10, 10)), iterations=3)
            thresh = cv2.bitwise_and(self.frame, self.frame, mask=thresh)
            cv2.imshow("motion", thresh)
            # Color range
            buffer = 20
            master_mask = np.zeros(thresh.shape[:2], dtype="uint8")
            # Color thresholding
            for color in self.colors:
                mask = cv2.inRange(thresh, (color - buffer).astype(int), (color + buffer).astype(int))
                master_mask = cv2.bitwise_or(mask, master_mask)
            # Fills the mask for easier detection
            master_mask = cv2.dilate(master_mask, np.ones((12, 12)), iterations=3)
            return master_mask
        else:
            return np.zeros((self.width, self.height), dtype="uint8")

    def manage_thresholding(self) -> None:
        """
        Modifies the state of the system based on time elapsed and the current state

        Returns:
            None
        """
        self.time_elapsed = int(time.time() - self.start_time)
        if self.time_elapsed >= self.timer_count:
            if self.state == States.COUNTDOWN_1:
                self.state = States.DONE_1
            elif self.state == States.COUNTDOWN_2:
                self.state = States.DONE_2
            self.get_color()

    def generate_contours(self, gray: np.array) -> List:
        """
        Generates a masked image with the generate_mask method and then computes the contours based on that and returns
        it

        Args:
            gray: Grayscale 'raw' image

        Returns:
            list of contours
        """
        mask = self.generate_mask(gray)
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = imutils.grab_contours(contours)
        drawn_contours = self.frame.copy()
        for c in contours:
            if cv2.contourArea(c) < 200:
                continue
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            for i in range(1, 5):
                if (self.width // 4) * (i - 1) < cX < (self.width // 4) * i:
                    if cY > self.height_thresh:
                        self.boxes[i - 1].has_collision()
                    else:
                        self.boxes[i - 1].reset_collision()
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(drawn_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("mask", mask)
        cv2.imshow("Contours", drawn_contours)
        return contours

    def webcam(self) -> None:
        """
        Main driver of the program, works out the timing and user inputs as well as calling the necessary
        methods.

        Returns:
            None
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        while True:
            if self.state in (States.COUNTDOWN_1, States.COUNTDOWN_2):
                self.manage_thresholding()
            _, frame = cap.read()
            frame = cv2.resize(frame, (700, 700))
            self.frame = cv2.flip(frame, 1)
            processed = self.impose_blocks()
            # Grayscale converted image of 'raw' image
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            # If color is confirmed, perform masking and contour detection
            if self.state == States.CONFIRM:
                contours = self.generate_contours(gray)
                with open("colors.pickle", "wb") as f:
                    pickle.dump(self.colors, f)
            # Otherwise prompt the user to calibrate
            else:
                processed = self.prompt_color()
            c = cv2.waitKey(1)
            if c == ord("q"):
                break
            # If space is pressed and the user is prompted to calibrate
            elif c == ord(" ") and self.state not in (
                    States.COUNTDOWN_1, States.COUNTDOWN_2, States.CONFIRM, States.DONE_2):
                if self.state == States.INIT:
                    self.state = States.COUNTDOWN_1
                elif self.state == States.DONE_1:
                    self.state = States.COUNTDOWN_2
                self.start_time = time.time()
            # If space is pressed and the user is prompted to confirm
            elif c == ord(" ") and self.state == States.DONE_2:
                self.state = States.CONFIRM
            # If escape is pressed and the user is prompted to confirm
            elif c == 27 and self.state == States.DONE_2:
                self.state = States.INIT
                self.colors.clear()
            # Populates frame queue for motion estimation
            self.frame_queue.append(gray.copy())
            if len(self.frame_queue) > 3:
                self.frame_queue.pop(0)
            cv2.imshow('Input', processed)

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Virtual Drums')
    parser.add_argument("-prev", action="store_true", help="Use previously stored colors")
    args = parser.parse_args()
    detector = Detector(args.prev)


if __name__ == "__main__":
    main()
