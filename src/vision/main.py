#!/usr/bin/env python3
import os
from typing import List
import time
from enum import Enum
import pickle
import argparse
import threading

import numpy as np
import imutils
import cv2
from pydub import AudioSegment
from pydub.playback import play


class States(Enum):
    """
    Enum class to facilitate program FSM
    """
    INIT = 0
    COUNTDOWN_1 = 1
    DONE_1 = 2
    CONFIRM_1 = 3
    COUNTDOWN_2 = 4
    DONE_2 = 5
    CONFIRM_2 = 6


class CollisionBox:
    """
    Collision box object made to be used in composition with the Detector class
    """

    def __init__(self, song_file: str, color: np.array, bias: float = 0):
        """
        Initializes collision box
        Args:
            song_file: Audio file name
            color: Respective color of collision box
            bias: Sound bias in decibels to control the volume in post
        """
        self.last_collision = 0
        self.song = AudioSegment.from_mp3(song_file) + bias
        self.color = np.array(color)
        self.velocity_timer = 0

    def has_collision(self) -> None:
        """
        Checks if enough time has expired to play sound

        Calculates the velocity of the stick during contact and adjusts the volume and pitch accordingly. Then plays
        the modified sound in a new thread to allow for non-blocking sound.

        Returns:
            None
        """
        if float(time.time() - self.last_collision) > 0.5:
            velocity = 56 / (time.time() - self.velocity_timer)
            # Volume calculation
            scale = round(velocity / 2000 * 25)
            scale = min(25, scale)
            scale = max(12, scale)

            # Pitch calculation
            octaves = round(velocity / 1400 * 1)
            octaves = min(1, octaves)
            octaves = max(0, octaves)
            octaves -= 0.5
            new_sample_rate = int(self.song.frame_rate * (2.0 ** octaves))
            pitch_modulated = self.song._spawn(self.song.raw_data, overrides={'frame_rate': new_sample_rate})

            self.last_collision = time.time()
            t = threading.Thread(target=play, args=(pitch_modulated + scale,))
            t.start()

    def reset_collision(self) -> None:
        """
        Sets the collision timer such that the next collision will play a sound.

        Is generally called when the stick has been detected *above* a certain threshold on the screen, signifying that
        the stick has reset itself.

        Returns:
            None
        """
        self.last_collision = time.time() - 1

    def start_timer(self) -> None:
        """
        Starts a timeout timer to avoid repeated hits.

        Returns:
            None
        """
        self.velocity_timer = time.time()


class Detector:
    """Drum detector and simulator

    Facilitates all of the computation behind tracking the drumsticks and calling the necessary functions to play sounds
    when a collision is detected.

    """

    def __init__(self, prev: bool):
        """
        Initializes Detector class

        Args:
            prev: Determines whether to use the previously detected color from the last run
        """
        self.height = 700
        self.width = 700
        # Seconds to wait for calibration
        self.timer_count = 2
        # Frame to be used for vision techniques
        self.frame = None
        # Frame to be shown to the user
        self.processed = None
        # Grayscale frame used in queue for motion detection
        self.gray = None
        self.frame_queue = []
        self.time_elapsed = 0
        self.start_time = 0
        self.colors = []
        if not prev:
            self.state = States.INIT
        else:
            self.state = States.CONFIRM_2
            with open("colors.pickle", "rb") as f:
                self.colors = pickle.load(f)
        self.circle_coordinates = [(int(self.width * 0.1), int(self.height * 0.93)),
                                   (int(self.width * 0.93), int(self.height * 0.93)),
                                   (int(self.width * 0.35), int(self.height * 0.93)),
                                   (int(self.width * 0.62), int(self.height * 0.93))]
        self.speed_thresh = self.height * 0.75
        self.reset_thresh = self.height * 0.78
        self.height_thresh = self.height * 0.83
        self.distance = self.height_thresh - self.speed_thresh
        prefix_dir = "../../resources/sounds/"
        self.boxes = [CollisionBox(os.path.join(prefix_dir, "drum1.mp3"), (255, 0, 0)),
                      CollisionBox(os.path.join(prefix_dir, "drum3.mp3"), (0, 255, 0), bias=5),
                      CollisionBox(os.path.join(prefix_dir, "drum2.mp3"), (0, 0, 255), bias=3),
                      CollisionBox(os.path.join(prefix_dir, "drum4.mp3"), (255, 0, 255), bias=5)]
        self.webcam()

    def measure_color(self) -> None:
        """
        Measures the average color in each respective detection circle

        Takes the average of a 4x4 square inside the circle for color detection to be used later.

        Returns:
            None
        """
        if self.state == States.DONE_1:
            coordinates = self.circle_coordinates[:2]
        elif self.state == States.DONE_2:
            coordinates = self.circle_coordinates[2:]
        else:
            coordinates = []
        for coord in coordinates:
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
        if self.state in (States.INIT, States.COUNTDOWN_1, States.COUNTDOWN_2):
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
        elif self.state in (States.DONE_1, States.DONE_2):
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
        self.processed = frame

    def draw_drums(self) -> np.array:
        """
        Draws bounding blocks on the bottom of the screen as different drum components

        Returns:
            Output frame
        """

        top_drum_y = int(0.875 * self.height)
        drum_width = self.width // 4
        frame = self.frame.copy()
        cv2.line(frame, (0, int(self.speed_thresh)), (self.width, int(self.speed_thresh)), (255, 0, 0), 2)
        cv2.line(frame, (0, int(self.reset_thresh)), (self.width, int(self.reset_thresh)), (255, 0, 0), 2)
        cv2.line(frame, (0, int(self.height_thresh)), (self.width, int(self.height_thresh)), (255, 0, 0), 2)
        for i in range(4):
            color_scale = 1.0 if float(time.time() - self.boxes[i].last_collision) < 0.2 else 0.7
            new_val = (frame[top_drum_y:, drum_width * i:drum_width * (i + 1), :] + self.boxes[
                i].color * color_scale) / 2
            frame[top_drum_y:, drum_width * i:drum_width * (i + 1), :] = new_val
        frame[frame > 255] = 255
        self.processed = frame

    def _generate_mask(self) -> np.array:
        """
        Uses motion estimation then color thresholding to create a mask to be used for contour detection

        Returns:
            Output mask
        """
        if self.frame_queue:
            # Motion estimation
            delta = cv2.absdiff(self.frame_queue[0], self.gray)
            movement_mask = cv2.threshold(delta, 20, 255, cv2.THRESH_BINARY)[1]
            movement_mask = cv2.dilate(movement_mask, np.ones((15, 15)), iterations=1)
            movement_output = cv2.bitwise_and(self.frame, self.frame, mask=movement_mask)
            cv2.imshow("motion", movement_output)
            # Color range
            buffer = 35
            master_mask = np.zeros(movement_output.shape[:2], dtype="uint8")

            # Color thresholding
            for color in self.colors:
                primary_index = np.argmax(color)
                bottom_range = color - buffer
                bottom_range[primary_index] += 20
                top_range = color + buffer
                top_range[primary_index] -= 20

                mask = cv2.inRange(movement_output, bottom_range.astype(int), top_range.astype(int))
                master_mask = cv2.bitwise_or(mask, master_mask)
            # Fills the mask for easier detection
            cv2.imshow("pre", master_mask)
            master_mask = cv2.erode(master_mask, np.ones((3, 3)), iterations=1)
            cv2.imshow("after-erosion", master_mask)
            master_mask = cv2.dilate(master_mask, np.ones((9, 9)), iterations=5)
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
            self.measure_color()

    def generate_contours(self) -> List:
        """
        Generates a masked image with the generate_mask method and then computes the contours based on that and returns
        it

        Returns:
            list of contours
        """
        mask = self._generate_mask()
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = imutils.grab_contours(contours)
        drawn_contours = self.frame.copy()
        contours = sorted(contours, key=lambda contour: cv2.contourArea(contour))
        for c in contours[-3:]:
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(drawn_contours, (cX, cY), 2, (0, 0, 255), 1)
            top_point = c[c[:, :, 1].argmin()][0, 1]
            for i in range(1, 5):
                if (self.width // 4) * (i - 1) < cX < (self.width // 4) * i:
                    if cY > self.height_thresh:
                        self.boxes[i - 1].has_collision()
                    else:
                        self.boxes[i - 1].reset_collision()
                    if top_point < self.speed_thresh:
                        self.boxes[i - 1].start_timer()
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
            self.draw_drums()
            # Grayscale converted image of 'raw' image
            self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            # If color is confirmed, perform masking and contour detection
            if self.state == States.CONFIRM_2:
                contours = self.generate_contours()
            # Otherwise prompt the user to calibrate
            else:
                self.prompt_color()
            c = cv2.waitKey(1)
            if c == ord("q"):
                break
            # If space is pressed and the user is prompted to calibrate
            elif c == ord(" ") and self.state not in (
                    States.COUNTDOWN_1, States.COUNTDOWN_2, States.CONFIRM_2, States.DONE_2):
                if self.state == States.INIT:
                    self.state = States.COUNTDOWN_1
                elif self.state == States.DONE_1:
                    self.state = States.COUNTDOWN_2
                self.start_time = time.time()
            # If space is pressed and the user is prompted to confirm
            elif c == ord(" ") and self.state in (States.DONE_1, States.DONE_2):
                if self.state == States.DONE_1:
                    self.state = States.CONFIRM_1
                else:
                    self.state = States.CONFIRM_2
                with open("colors.pickle", "wb") as f:
                    pickle.dump(self.colors, f)
            # If escape is pressed and the user is prompted to confirm
            elif c == 27 and self.state in (States.DONE_1, States.DONE_2):
                self.state = States.INIT
                self.colors.clear()
            # Populates frame queue for motion estimation
            self.frame_queue.append(self.gray.copy())
            if len(self.frame_queue) > 3:
                self.frame_queue.pop(0)
            cv2.imshow('Input', self.processed)

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Virtual Drums')
    parser.add_argument("-prev", action="store_true", help="Use previously stored colors")
    args = parser.parse_args()
    detector = Detector(args.prev)


if __name__ == "__main__":
    main()
